"""
STEP 5 + 6 — Hybrid retrieval + reranking.

Flow:
   query ──┬──> vector search (meaning)  ── top 20 ──┐
           └──> BM25 search  (keywords) ── top 20 ──┴──> RRF fusion ──> cross-encoder rerank ──> top 5

RRF (Reciprocal Rank Fusion): har list mein chunk ki RANK dekho, score = sum(1 / (k + rank)).
Fayda: vector scores (0-1) aur BM25 scores (0-30) alag scale pe hote hain; RRF sirf rank use
karta hai, to normalize karne ka jhanjhat nahi. Jo chunk dono lists mein upar hai, woh jeetta hai.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .bm25 import BM25
from .chunker import Chunk
from .vector_store import VectorStore


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float                       # final score (rerank score agar reranker on hai, warna RRF)
    vector_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    debug: Dict[str, float] = field(default_factory=dict)


def reciprocal_rank_fusion(rank_lists: List[List[int]], k: int = 60) -> Dict[int, float]:
    fused: Dict[int, float] = {}
    for ranking in rank_lists:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


class HybridRetriever:
    def __init__(self, store: VectorStore, embedder, reranker=None, settings=None):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.s = settings
        self._bm25: Optional[BM25] = None
        self._bm25_version = -1

    @property
    def bm25(self) -> BM25:
        # Store badla to BM25 dobara banao (sasta hai — sirf tokenization)
        if self._bm25 is None or self._bm25_version != self.store.version:
            self._bm25 = BM25([c.text for c in self.store.chunks])
            self._bm25_version = self.store.version
        return self._bm25

    def retrieve(self, query: str, top_k: Optional[int] = None, source_filter: Optional[List[str]] = None) -> List[RetrievedChunk]:
        s = self.s
        top_k = top_k or s.final_top_k
        if not len(self.store):
            return []

        allowed = None
        if source_filter:
            allowed = {i for i, c in enumerate(self.store.chunks) if c.metadata.get("source") in set(source_filter)}

        # 1) Vector search (filter ho to zyada candidates lo, phir filter karo)
        qvec = self.embedder.embed_query(query)
        fetch = s.vector_top_k * (4 if allowed is not None else 1)
        vec_hits = [(i, sc) for i, sc in self.store.search(qvec, fetch) if allowed is None or i in allowed][: s.vector_top_k]

        # 2) BM25 search
        bm_hits = [(i, sc) for i, sc in self.bm25.top_k(query, s.bm25_top_k * (4 if allowed is not None else 1))
                   if allowed is None or i in allowed][: s.bm25_top_k]

        vec_rank = {i: r for r, (i, _) in enumerate(vec_hits, start=1)}
        bm_rank = {i: r for r, (i, _) in enumerate(bm_hits, start=1)}
        vec_score = dict(vec_hits)
        bm_score = dict(bm_hits)

        # 3) Fusion
        fused = reciprocal_rank_fusion([[i for i, _ in vec_hits], [i for i, _ in bm_hits]], k=s.rrf_k)
        candidates = sorted(fused, key=fused.get, reverse=True)

        results = [
            RetrievedChunk(
                chunk=self.store.chunks[i],
                score=fused[i],
                vector_rank=vec_rank.get(i),
                bm25_rank=bm_rank.get(i),
                debug={"rrf": fused[i], "cosine": vec_score.get(i, 0.0), "bm25": bm_score.get(i, 0.0)},
            )
            for i in candidates
        ]

        # 4) Rerank
        if self.reranker is not None and results:
            rr = self.reranker.score(query, [r.chunk.text for r in results])
            for r, sc in zip(results, rr):
                r.score = sc
                r.debug["rerank"] = sc
            results.sort(key=lambda r: r.score, reverse=True)
            results = [r for r in results if r.score >= s.rerank_min_score]

        return results[:top_k]
