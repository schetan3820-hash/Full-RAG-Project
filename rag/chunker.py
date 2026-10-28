"""
STEP 2 — Chunking.

Recursive splitting: pehle paragraph (\n\n) pe todne ki koshish, phir line (\n),
phir sentence (". "), phir word (" ") — taaki text natural jagah pe toote.
Overlap rakhte hain taaki do chunks ke beech mein toota hua context bach jaaye.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List

from .loaders import Document

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)

    def citation(self) -> str:
        src = self.metadata.get("source", "unknown")
        if "page" in self.metadata:
            return f"{src}, page {self.metadata['page']}"
        if "row" in self.metadata:
            return f"{src}, row {self.metadata['row']}"
        return src


class RecursiveChunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150, separators: List[str] | None = None):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap chunk_size se chhota hona chahiye")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or DEFAULT_SEPARATORS

    # ---------- public ----------
    def split_text(self, text: str) -> List[str]:
        return [c.strip() for c in self._split(text, self.separators) if c.strip()]

    def split_documents(self, documents: List[Document]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for doc in documents:
            for i, piece in enumerate(self.split_text(doc.text)):
                meta = dict(doc.metadata)
                meta["chunk_index"] = i
                key = f"{meta.get('path', meta.get('source', ''))}|{meta.get('page', '')}|{meta.get('row', '')}|{i}|{piece}"
                cid = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
                chunks.append(Chunk(id=cid, text=piece, metadata=meta))
        return chunks

    # ---------- internals ----------
    def _split(self, text: str, separators: List[str]) -> List[str]:
        # Pehla separator chuno jo text mein maujood hai
        sep = separators[-1]
        rest: List[str] = []
        for i, s in enumerate(separators):
            if s == "" or s in text:
                sep, rest = s, separators[i + 1 :]
                break

        pieces = text.split(sep) if sep else list(text)
        final: List[str] = []
        good: List[str] = []
        for piece in pieces:
            if len(piece) <= self.chunk_size:
                good.append(piece)
            else:
                if good:
                    final.extend(self._merge(good, sep))
                    good = []
                if rest:
                    final.extend(self._split(piece, rest))
                else:
                    final.append(piece)
        if good:
            final.extend(self._merge(good, sep))
        return final

    def _merge(self, splits: List[str], sep: str) -> List[str]:
        """Chhote pieces ko jod ke ~chunk_size ke chunks banao, overlap ke saath."""
        docs: List[str] = []
        current: List[str] = []
        total = 0
        sep_len = len(sep)
        for s in splits:
            s_len = len(s)
            extra = sep_len if current else 0
            if total + s_len + extra > self.chunk_size and current:
                docs.append(sep.join(current))
                # Overlap: aage se pieces hatao jab tak bacha hua part overlap se chhota na ho
                while current and (
                    total > self.chunk_overlap or (total + s_len + (sep_len if current else 0) > self.chunk_size)
                ):
                    total -= len(current[0]) + (sep_len if len(current) > 1 else 0)
                    current.pop(0)
            current.append(s)
            total += s_len + (sep_len if len(current) > 1 else 0)
        if current:
            docs.append(sep.join(current))
        return docs
