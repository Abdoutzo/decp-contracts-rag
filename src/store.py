"""FAISS vector store.

Embeddings are L2-normalized, so inner product == cosine similarity.
IndexFlatIP is exact search; fine for corpora this size. If you ever
index the full DECP dump, switch to IVF and thank me later.
"""
from dataclasses import dataclass
from typing import List, Tuple

import faiss
import numpy as np

from src.ingest import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: List[Chunk] = []

    def add(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        assert embeddings.shape[1] == self.dim
        self.index.add(np.ascontiguousarray(embeddings.astype(np.float32)))
        self.chunks.extend(chunks)

    def search(self, query_emb: np.ndarray, k: int) -> List[Hit]:
        q = np.ascontiguousarray(query_emb.reshape(1, -1).astype(np.float32))
        scores, idxs = self.index.search(q, min(k, len(self.chunks)))
        return [
            Hit(chunk=self.chunks[i], score=float(s))
            for s, i in zip(scores[0], idxs[0]) if i != -1
        ]

    def doc_ids(self, hits: List[Hit]) -> List[str]:
        seen, out = set(), []
        for h in hits:
            if h.chunk.doc_id not in seen:
                seen.add(h.chunk.doc_id)
                out.append(h.chunk.doc_id)
        return out

    def __len__(self) -> int:
        return len(self.chunks)
