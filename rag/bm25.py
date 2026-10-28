"""
Keyword search — BM25 (Okapi).

Vector search "meaning" pakadti hai, lekin exact words (naam, codes, IDs, acronyms
jaise "GATE 2027", "LoRA", "Section 302") mein kamzor hoti hai. BM25 exact-word match
mein strong hai. Dono ko milake hybrid search banta hai.

Score ka idea:
  - Jo word query mein hai aur chunk mein baar-baar aata hai -> score badhta hai (TF)
  - Jo word poore corpus mein rare hai, uska weight zyada (IDF)
  - Lambe chunks ko thoda penalize karo (length normalization, parameter b)
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Tuple

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "to", "in", "on", "for",
    "and", "or", "with", "as", "by", "at", "it", "this", "that", "from", "what", "which",
    "how", "why", "who", "do", "does", "did", "can", "i", "you", "we", "they", "me", "my",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class BM25:
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tokens = [tokenize(t) for t in corpus]
        self.doc_freqs = [Counter(toks) for toks in self.doc_tokens]
        self.doc_len = [len(toks) for toks in self.doc_tokens]
        self.n_docs = len(corpus)
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0

        df: Counter = Counter()
        for freqs in self.doc_freqs:
            df.update(freqs.keys())
        # +1 wala IDF variant: kabhi negative nahi hota
        self.idf = {term: math.log((self.n_docs - n + 0.5) / (n + 0.5) + 1.0) for term, n in df.items()}

        # Inverted index: term -> jin docs mein hai (fast scoring)
        self.postings: dict = {}
        for i, freqs in enumerate(self.doc_freqs):
            for term in freqs:
                self.postings.setdefault(term, []).append(i)

    def scores(self, query: str) -> dict:
        out: dict = {}
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i in self.postings[term]:
                tf = self.doc_freqs[i][term]
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1))
                out[i] = out.get(i, 0.0) + idf * tf * (self.k1 + 1) / denom
        return out

    def top_k(self, query: str, k: int) -> List[Tuple[int, float]]:
        s = self.scores(query)
        return sorted(s.items(), key=lambda x: x[1], reverse=True)[:k]
