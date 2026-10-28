"""
STEP 6 — Reranking (cross-encoder).

Bi-encoder (embedding model) query aur chunk ko ALAG-ALAG vector banata hai -> fast, lekin approx.
Cross-encoder query + chunk ko EK SAATH padhta hai aur relevance score deta hai -> slow, lekin
kaafi zyada accurate. Isliye: pehle hybrid search se ~30 candidates nikalo (fast),
phir sirf unpe cross-encoder chalao aur best 5 rakho. Ye RAG quality ka sabse bada boost hai.
"""
from __future__ import annotations

from typing import List


class Reranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, query: str, texts: List[str]) -> List[float]:
        if not texts:
            return []
        scores = self.model.predict([(query, t) for t in texts])
        return [float(s) for s in scores]
