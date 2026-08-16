"""Thin wrapper around sentence-transformers.

Isolated in its own module so chunking.py and retrieval/dense.py depend on a
narrow `embed(texts) -> np.ndarray` interface rather than the
sentence-transformers API directly - swapping the embedding model (e.g. for
the ablation study, or a larger model later) only touches this file.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class SentenceEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Return an (n_texts, dim) float32 array of embeddings."""
        if not texts:
            return np.zeros((0, self._model.get_sentence_embedding_dimension()), dtype=np.float32)
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()
