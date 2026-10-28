"""
STEP 4 — Vector store (disk pe persist hota hai).

Kya store hota hai (storage/ folder mein):
  - embeddings.npy  : saare chunk vectors (normalized)
  - chunks.jsonl    : har chunk ka id, text, metadata (source, page...)
  - manifest.json   : kaunsi file kis hash ke saath ingest hui (incremental update ke liye)

Search: FAISS (IndexFlatIP = exact inner-product search) agar installed hai,
warna numpy matrix multiplication. Vectors normalized hain, to inner product = cosine.

Features jo tutorial wale code mein nahi the:
  - metadata (source/page) save hota hai -> citations possible
  - incremental add: nayi file aaye to poora index dobara nahi banana
  - file badli to purane chunks delete + naye add (koi duplicate nahi)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .chunker import Chunk

try:
    import faiss  # type: ignore

    _HAS_FAISS = True
except ImportError:  # numpy fallback
    _HAS_FAISS = False


class VectorStore:
    def __init__(self, storage_dir: Path):
        self.dir = Path(storage_dir)
        self.chunks: List[Chunk] = []
        self.embeddings: np.ndarray | None = None
        self.manifest: Dict[str, dict] = {}  # path -> {"hash":..., "n_chunks":..., "source":...}
        self.version = 0                     # har change pe badhta hai (BM25 rebuild ke liye)
        self._index = None
        self.load()

    # ---------------- persistence ----------------
    @property
    def _paths(self):
        return self.dir / "embeddings.npy", self.dir / "chunks.jsonl", self.dir / "manifest.json"

    def load(self):
        emb_p, chunks_p, man_p = self._paths
        if not (emb_p.exists() and chunks_p.exists()):
            return
        self.embeddings = np.load(emb_p).astype("float32")
        with open(chunks_p, encoding="utf-8") as f:
            self.chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
        if man_p.exists():
            self.manifest = json.loads(man_p.read_text(encoding="utf-8"))
        if len(self.chunks) != len(self.embeddings):
            raise RuntimeError("Storage corrupt hai (chunks aur embeddings ki count alag). `python ingest.py --reset` chalao.")
        self._rebuild_index()

    def save(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        emb_p, chunks_p, man_p = self._paths
        if self.embeddings is None:
            self.embeddings = np.zeros((0, 0), dtype="float32")
        np.save(emb_p, self.embeddings)
        with open(chunks_p, "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps({"id": c.id, "text": c.text, "metadata": c.metadata}, ensure_ascii=False) + "\n")
        man_p.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")

    def reset(self):
        self.chunks, self.embeddings, self.manifest = [], None, {}
        self._index = None
        self.version += 1
        for p in self._paths:
            if p.exists():
                p.unlink()

    # ---------------- write ----------------
    def add(self, chunks: List[Chunk], embeddings: np.ndarray):
        if not chunks:
            return
        embeddings = np.asarray(embeddings, dtype="float32")
        if self.embeddings is None or self.embeddings.size == 0:
            self.embeddings = embeddings
        else:
            if embeddings.shape[1] != self.embeddings.shape[1]:
                raise ValueError(
                    "Embedding dimension match nahi kar raha — shayad embedding model badla hai. "
                    "`python ingest.py --reset` chalao."
                )
            self.embeddings = np.vstack([self.embeddings, embeddings])
        self.chunks.extend(chunks)
        self._rebuild_index()

    def delete_path(self, path: str) -> int:
        keep = [i for i, c in enumerate(self.chunks) if c.metadata.get("path") != path]
        removed = len(self.chunks) - len(keep)
        if removed:
            self.chunks = [self.chunks[i] for i in keep]
            self.embeddings = self.embeddings[keep] if keep else None
            self._rebuild_index()
        self.manifest.pop(path, None)
        return removed

    def _rebuild_index(self):
        self.version += 1
        self._index = None
        if _HAS_FAISS and self.embeddings is not None and len(self.embeddings):
            self._index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self._index.add(self.embeddings)

    # ---------------- read ----------------
    def __len__(self):
        return len(self.chunks)

    def search(self, query_vec: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """Returns [(chunk_position, cosine_score), ...] best pehle."""
        if not len(self.chunks):
            return []
        k = min(k, len(self.chunks))
        q = np.asarray(query_vec, dtype="float32").reshape(1, -1)
        if self._index is not None:
            scores, idx = self._index.search(q, k)
            return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i != -1]
        sims = self.embeddings @ q[0]
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(i), float(sims[i])) for i in top]

    def sources(self) -> List[dict]:
        return [{"path": p, **info} for p, info in sorted(self.manifest.items())]
