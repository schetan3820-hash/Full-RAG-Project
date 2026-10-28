"""
Tests — fake (chhote) models ke saath, taaki bina internet / GPU / API key ke chal sakein.

  python -m pytest tests -q      (ya)   python tests/test_rag.py
"""
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.bm25 import BM25, tokenize  # noqa: E402
from rag.chunker import RecursiveChunker  # noqa: E402
from rag.config import Settings  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402
from rag.prompts import NO_CONTEXT_ANSWER  # noqa: E402
from rag.retriever import reciprocal_rank_fusion  # noqa: E402


# ---------------- fakes ----------------
class FakeEmbedder:
    """Hashed bag-of-words — similar words wale texts ke vectors paas aate hain."""
    dim = 256

    def _vec(self, text):
        v = np.zeros(self.dim, dtype="float32")
        for t in tokenize(text):
            v[hash(t) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


class FakeReranker:
    def score(self, query, texts):
        q = set(tokenize(query))
        return [len(q & set(tokenize(t))) - 0.5 for t in texts]


class FakeLLM:
    def __init__(self):
        self.calls = []

    def chat(self, messages, temperature=None, json_mode=False):
        self.calls.append(messages)
        content = messages[-1]["content"]
        if "STANDALONE QUERY" in content:
            return "What is the overlap in chunking?"
        return "Answer based on context [1]."

    def stream(self, messages):
        self.calls.append(messages)
        for w in ["Streamed ", "answer ", "[1]."]:
            yield w


def make_pipeline(tmp: Path, **overrides):
    s = Settings()
    s.data_dir = tmp / "data"
    s.storage_dir = tmp / "storage"
    s.chunk_size, s.chunk_overlap = 300, 50
    s.rerank_min_score = 0.0
    for k, v in overrides.items():
        setattr(s, k, v)
    return RAGPipeline(settings=s, embedder=FakeEmbedder(), reranker=FakeReranker(), llm=FakeLLM())


# ---------------- unit tests ----------------
def test_chunker_respects_size_and_overlap():
    text = " ".join(f"Sentence number {i} talks about topic {i % 7}." for i in range(200))
    ch = RecursiveChunker(chunk_size=200, chunk_overlap=40)
    chunks = ch.split_text(text)
    assert len(chunks) > 10
    assert all(len(c) <= 200 for c in chunks)
    # overlap: consecutive chunks share some text
    shared = sum(1 for a, b in zip(chunks, chunks[1:]) if a.split()[-1] in b)
    assert shared >= len(chunks) // 2
    # nothing lost: every sentence number appears somewhere
    joined = " ".join(chunks)
    for i in range(200):
        assert f"Sentence number {i} " in joined


def test_chunker_prefers_paragraphs():
    text = "Para one is short.\n\nPara two is also short.\n\n" + "x" * 50
    ch = RecursiveChunker(chunk_size=30, chunk_overlap=5)
    chunks = ch.split_text(text)
    assert chunks[0] == "Para one is short."
    assert chunks[1] == "Para two is also short."


def test_chunker_huge_word():
    ch = RecursiveChunker(chunk_size=50, chunk_overlap=10)
    chunks = ch.split_text("a" * 500)
    assert all(len(c) <= 50 for c in chunks) and len("".join(chunks)) >= 500


def test_bm25_ranks_exact_keyword_first():
    corpus = ["LoRA is a parameter efficient finetuning method", "Cats are cute animals", "Transformers use attention"]
    bm = BM25(corpus)
    top = bm.top_k("what is LoRA", 3)
    assert top[0][0] == 0
    assert all(i != 1 for i, _ in top)  # unrelated doc not scored


def test_rrf_prefers_docs_in_both_lists():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 4, 1]], k=60)
    best = max(fused, key=fused.get)
    assert best in (1, 3)
    assert fused[1] > fused[2] and fused[3] > fused[4]


# ---------------- integration tests ----------------
def test_end_to_end_ingest_ask_and_incremental():
    tmp = Path(tempfile.mkdtemp())
    try:
        data = tmp / "data"
        data.mkdir()
        shutil.copy(ROOT / "data" / "rag_notes.md", data / "rag_notes.md")
        (data / "cooking.txt").write_text("To make tea, boil water, add tea leaves, milk and sugar. Masala chai uses ginger and cardamom.")

        rag = make_pipeline(tmp)
        stats = rag.ingest(verbose=False)
        assert sorted(stats["added"]) == ["cooking.txt", "rag_notes.md"]
        n = len(rag.store)
        assert n > 5

        # retrieval finds the right chunk
        hits = rag.retriever.retrieve("Reciprocal Rank Fusion rank formula", top_k=3)
        assert hits and "Reciprocal Rank Fusion" in hits[0].chunk.text
        assert hits[0].chunk.metadata["source"] == "rag_notes.md"

        # source filter
        hits = rag.retriever.retrieve("tea ginger", top_k=3, source_filter=["cooking.txt"])
        assert hits and all(h.chunk.metadata["source"] == "cooking.txt" for h in hits)

        # ask -> answer + numbered sources + context in prompt
        res = rag.ask("How does reranking with a cross-encoder work?")
        assert res.answer.startswith("Answer")
        assert res.sources and res.sources[0].n == 1
        user_msg = rag.llm.calls[-1][-1]["content"]
        assert "[1] (Source: rag_notes.md)" in user_msg

        # irrelevant question -> reranker filters everything -> no-context answer, no LLM call
        before = len(rag.llm.calls)
        res = rag.ask("zebra quantum pizza")
        assert res.answer == NO_CONTEXT_ANSWER and not res.sources
        assert len(rag.llm.calls) == before

        # history -> query rewrite
        res = rag.ask("and why overlap?", history=[{"role": "user", "content": "explain chunking"},
                                                   {"role": "assistant", "content": "Chunking splits docs."}])
        assert res.search_query == "What is the overlap in chunking?"

        # streaming
        events = list(rag.ask_stream("What is BM25?"))
        assert events[0]["type"] == "sources" and events[0]["sources"]
        assert "".join(e["text"] for e in events[1:]) == "Streamed answer [1]."

        # unchanged -> skipped
        stats = rag.ingest(verbose=False)
        assert sorted(stats["skipped"]) == ["cooking.txt", "rag_notes.md"] and stats["chunks_added"] == 0
        assert len(rag.store) == n

        # modified -> updated, no duplicates
        (data / "cooking.txt").write_text("Coffee: brew ground beans with hot water.")
        stats = rag.ingest(verbose=False)
        assert stats["updated"] == ["cooking.txt"]
        assert not any("Masala chai" in c.text for c in rag.store.chunks)
        assert any("Coffee" in c.text for c in rag.store.chunks)

        # persistence: new pipeline loads same index
        rag2 = make_pipeline(tmp)
        assert len(rag2.store) == len(rag.store)
        assert rag2.retriever.retrieve("coffee beans", top_k=1)[0].chunk.metadata["source"] == "cooking.txt"

        # deleted file -> removed
        (data / "cooking.txt").unlink()
        stats = rag2.ingest(verbose=False)
        assert stats["removed"] == ["cooking.txt"]
        assert all(c.metadata["source"] != "cooking.txt" for c in rag2.store.chunks)

        # reset
        rag2.reset()
        assert len(rag2.store) == 0
        assert rag2.ask("anything").answer == NO_CONTEXT_ANSWER
    finally:
        shutil.rmtree(tmp)


def test_pdf_and_docx_loading_with_pages():
    from rag.loaders import load_file

    tmp = Path(tempfile.mkdtemp())
    try:
        try:
            import docx

            d = docx.Document()
            d.add_paragraph("GATE syllabus includes algorithms and operating systems.")
            d.save(tmp / "g.docx")
            docs = load_file(tmp / "g.docx")
            assert "algorithms" in docs[0].text and docs[0].metadata["source"] == "g.docx"
        except ImportError:
            pass

        try:
            from reportlab.pdfgen import canvas
        except ImportError:
            canvas = None
        if canvas:
            c = canvas.Canvas(str(tmp / "t.pdf"))
            c.drawString(100, 750, "Page one text about LoRA")
            c.showPage()
            c.drawString(100, 750, "Page two text about QLoRA")
            c.save()
            docs = load_file(tmp / "t.pdf")
            assert [d.metadata["page"] for d in docs] == [1, 2]
            assert "QLoRA" in docs[1].text
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("✓", fn.__name__)
    print(f"\nAll {len(fns)} tests passed.")
