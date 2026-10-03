"""Embedding wrapper.

One class, one model, batching built in. I use a multilingual MPNet
because the corpus is French and I want decent French representations
without needing a GPU.
"""
from typing import List

import numpy as np

import config


class Embedder:
    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Return L2-normalized embeddings (cosine-ready)."""
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()))
        embs = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        return np.asarray(embs, dtype=np.float32)

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()
