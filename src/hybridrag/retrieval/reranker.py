"""Cross-encoder reranking: the second stage of a "retrieve wide, rerank
narrow" two-stage pipeline.

Why rerank at all, given hybrid retrieval already returns a ranked list:
dense embeddings and BM25 both score a query against a chunk independently
and compare the two resulting scores - fast enough to run over an entire
corpus, but the model never actually looks at the query and the chunk
together. A cross-encoder concatenates (query, chunk) and runs them through
one transformer forward pass, so attention can directly relate query tokens
to chunk tokens - a strictly more accurate relevance signal per pair, at the
cost of one transformer pass per candidate. That cost is why it's applied
only to the ~20 hybrid candidates rather than to the whole corpus: hybrid
retrieval is optimized for recall (don't miss the right chunk among
thousands), reranking is optimized for precision on the small set hybrid
retrieval already found.
"""

from __future__ import annotations

import logging
import time
from typing import Protocol

from hybridrag.types import ScoredChunk

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class _PredictsPairs(Protocol):
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_MODEL, model: _PredictsPairs | None = None) -> None:
        """`model` can be injected (any object with a `.predict(pairs)`
        method) so tests don't need to download a real cross-encoder - the
        same pattern DenseIndex uses for its embedder."""
        self.model_name = model_name
        if model is not None:
            self._model = model
        else:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(model_name)
        self.last_latency_seconds: float = 0.0

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int = 5) -> list[ScoredChunk]:
        """Score every (query, candidate) pair with the cross-encoder and
        return the top_k highest-scoring candidates.

        Measures and logs its own latency in isolation from retrieval, since
        that isolated number - not total pipeline time - is what tells you
        what reranking specifically costs.
        """
        if not candidates:
            return []

        pairs = [(query, sc.chunk.text) for sc in candidates]

        start = time.perf_counter()
        scores = self._model.predict(pairs)
        latency = time.perf_counter() - start
        self.last_latency_seconds = latency

        logger.info(
            "Reranked %d candidates -> top %d in %.1fms (%.2fms/candidate)",
            len(candidates),
            min(top_k, len(candidates)),
            latency * 1000,
            (latency * 1000) / len(candidates),
        )

        rescored = [
            ScoredChunk(chunk=sc.chunk, score=float(score)) for sc, score in zip(candidates, scores)
        ]
        rescored.sort(key=lambda sc: sc.score, reverse=True)
        return rescored[:top_k]
