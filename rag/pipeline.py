"""
Poora RAG ek class mein — RAGPipeline.

  ingest():  files -> load -> chunk -> embed -> store            (offline, ek baar)
  ask():     question -> (rewrite with history) -> hybrid retrieve -> rerank
             -> numbered context -> LLM -> answer + citations     (har query pe)
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

from .chunker import RecursiveChunker
from .config import Settings, settings as default_settings
from .loaders import file_hash, iter_files, load_file
from .prompts import (
    ANSWER_SYSTEM_PROMPT,
    ANSWER_USER_TEMPLATE,
    CONDENSE_PROMPT,
    NO_CONTEXT_ANSWER,
)
from .retriever import HybridRetriever, RetrievedChunk
from .vector_store import VectorStore


@dataclass
class Source:
    n: int
    citation: str
    source: str
    page: Optional[int]
    text: str
    score: float


@dataclass
class RAGAnswer:
    question: str
    search_query: str
    answer: str
    sources: List[Source] = field(default_factory=list)
    timings: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class RAGPipeline:
    def __init__(self, settings: Settings | None = None, embedder=None, reranker=None, llm=None):
        """embedder/reranker/llm inject kar sakte ho (tests mein fake models ke liye)."""
        self.s = settings or default_settings
        if embedder is None:
            from .embedder import Embedder

            embedder = Embedder(self.s.embedding_model, self.s.query_instruction)
        if reranker is None and self.s.use_reranker:
            from .reranker import Reranker

            reranker = Reranker(self.s.reranker_model)
        if llm is None:
            from .llm import LLM

            llm = LLM(self.s.llm_base_url, self.s.llm_api_key, self.s.llm_model, self.s.llm_temperature, self.s.llm_max_tokens)

        self.embedder, self.reranker, self.llm = embedder, reranker, llm
        self.store = VectorStore(self.s.storage_dir)
        self.chunker = RecursiveChunker(self.s.chunk_size, self.s.chunk_overlap)
        self.retriever = HybridRetriever(self.store, embedder, reranker, self.s)

    # ===================== INGESTION =====================
    def ingest(self, path: str | Path | None = None, force: bool = False, verbose: bool = True) -> dict:
        """
        Folder/file ingest karo. Incremental hai:
          - nayi file          -> add
          - badli hui file     -> purane chunks delete, naye add
          - same file (hash)   -> skip
          - data se hata di    -> uske chunks delete  (sirf jab poora data folder ingest ho)
        """
        root = Path(path or self.s.data_dir)
        if not root.exists():
            raise FileNotFoundError(f"Path nahi mila: {root}")
        stats = {"added": [], "updated": [], "skipped": [], "removed": [], "failed": {}, "chunks_added": 0}
        seen = set()

        for f in iter_files(root):
            key = str(f.resolve())
            seen.add(key)
            h = file_hash(f)
            old = self.store.manifest.get(key)
            if old and old.get("hash") == h and not force:
                stats["skipped"].append(f.name)
                continue
            try:
                docs = load_file(f)
                for d in docs:
                    d.metadata["path"] = key
                chunks = self.chunker.split_documents(docs)
                if not chunks:
                    stats["failed"][f.name] = "koi text nahi mila (scanned PDF? OCR chahiye)"
                    continue
                embs = self.embedder.embed_documents([c.text for c in chunks])
                if old:
                    self.store.delete_path(key)
                self.store.add(chunks, embs)
                self.store.manifest[key] = {"source": f.name, "hash": h, "n_chunks": len(chunks)}
                stats["updated" if old else "added"].append(f.name)
                stats["chunks_added"] += len(chunks)
                if verbose:
                    print(f"  ✓ {f.name}: {len(docs)} pages/sections -> {len(chunks)} chunks")
            except Exception as e:  # ek kharab file poora ingest na roke
                stats["failed"][f.name] = f"{type(e).__name__}: {e}"
                if verbose:
                    print(f"  ✗ {f.name}: {e}")

        if root.is_dir() and root.resolve() == Path(self.s.data_dir).resolve():
            for key in [k for k in list(self.store.manifest) if k not in seen]:
                name = self.store.manifest[key].get("source", key)
                self.store.delete_path(key)
                stats["removed"].append(name)

        self.store.save()
        stats["total_chunks"] = len(self.store)
        return stats

    def reset(self):
        self.store.reset()

    # ===================== QUERY =====================
    def _format_history(self, history: List[dict]) -> str:
        turns = history[-self.s.max_history_turns * 2 :]
        return "\n".join(f"{m['role'].upper()}: {m['content'][:800]}" for m in turns)

    def rewrite_query(self, question: str, history: Optional[List[dict]]) -> str:
        """Follow-up ("aur uska time complexity?") ko standalone query mein badlo."""
        if not history:
            return question
        prompt = CONDENSE_PROMPT.format(history=self._format_history(history), question=question)
        try:
            q = self.llm.chat([{"role": "user", "content": prompt}], temperature=0).strip().strip('"')
            return q or question
        except Exception:
            return question

    @staticmethod
    def build_context(hits: List[RetrievedChunk]) -> str:
        return "\n\n".join(f"[{i}] (Source: {h.chunk.citation()})\n{h.chunk.text}" for i, h in enumerate(hits, start=1))

    def _messages(self, question: str, hits: List[RetrievedChunk], history: Optional[List[dict]]) -> List[dict]:
        msgs = [{"role": "system", "content": ANSWER_SYSTEM_PROMPT}]
        for m in (history or [])[-self.s.max_history_turns * 2 :]:
            msgs.append({"role": m["role"], "content": m["content"]})
        msgs.append({"role": "user", "content": ANSWER_USER_TEMPLATE.format(context=self.build_context(hits), question=question)})
        return msgs

    @staticmethod
    def _sources(hits: List[RetrievedChunk]) -> List[Source]:
        return [
            Source(
                n=i,
                citation=h.chunk.citation(),
                source=h.chunk.metadata.get("source", ""),
                page=h.chunk.metadata.get("page"),
                text=h.chunk.text,
                score=round(float(h.score), 4),
            )
            for i, h in enumerate(hits, start=1)
        ]

    def retrieve(self, question: str, history=None, top_k=None, source_filter=None):
        t0 = time.perf_counter()
        search_query = self.rewrite_query(question, history)
        t1 = time.perf_counter()
        hits = self.retriever.retrieve(search_query, top_k=top_k, source_filter=source_filter)
        t2 = time.perf_counter()
        return search_query, hits, {"rewrite_s": round(t1 - t0, 3), "retrieve_s": round(t2 - t1, 3)}

    def ask(self, question: str, history: Optional[List[dict]] = None, top_k: Optional[int] = None,
            source_filter: Optional[List[str]] = None) -> RAGAnswer:
        search_query, hits, timings = self.retrieve(question, history, top_k, source_filter)
        if not hits:
            return RAGAnswer(question, search_query, NO_CONTEXT_ANSWER, [], timings)
        t0 = time.perf_counter()
        answer = self.llm.chat(self._messages(question, hits, history))
        timings["generate_s"] = round(time.perf_counter() - t0, 3)
        return RAGAnswer(question, search_query, answer.strip(), self._sources(hits), timings)

    def ask_stream(self, question: str, history: Optional[List[dict]] = None, top_k: Optional[int] = None,
                   source_filter: Optional[List[str]] = None) -> Iterator[dict]:
        """Yields {"type": "sources", ...} pehle, phir {"type": "token", "text": ...} tokens."""
        search_query, hits, timings = self.retrieve(question, history, top_k, source_filter)
        yield {"type": "sources", "search_query": search_query, "sources": [asdict(s) for s in self._sources(hits)]}
        if not hits:
            yield {"type": "token", "text": NO_CONTEXT_ANSWER}
            return
        for tok in self.llm.stream(self._messages(question, hits, history)):
            yield {"type": "token", "text": tok}
