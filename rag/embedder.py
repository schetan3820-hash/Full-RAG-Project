"""
STEP 3 — Embeddings.

Text -> vector (numbers ki list). Similar meaning wale texts ke vectors paas-paas hote hain.
Default model: BAAI/bge-small-en-v1.5 — free, local chalta hai, 384-dim, aur
all-MiniLM-L6-v2 se retrieval mein behtar hai.

Vectors ko normalize karte hain (length = 1), taaki dot product = cosine similarity ho.
"""
from __future__ import annotations

from typing import List

import numpy as np


class Embedder:
    def __init__(self, model_name: str, query_instruction: str = ""):
        self.model_name = model_name
        self.query_instruction = query_instruction
        self._model = None

    @property
    def model(self):
        # Lazy load: model tabhi load ho jab pehli baar zarurat ho (startup fast rehta hai)
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        vecs = self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=len(texts) > 64
        )
        return np.asarray(vecs, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        # BGE jaise models query ke aage instruction chahte hain — documents ke aage nahi
        vec = self.model.encode([self.query_instruction + text], normalize_embeddings=True)
        return np.asarray(vec, dtype="float32")[0]
