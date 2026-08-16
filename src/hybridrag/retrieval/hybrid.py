"""Combine dense and sparse retrieval via reciprocal rank fusion.

Concrete motivation from this corpus (not a hypothetical): in the Phase 1
demo, the query "What is the time complexity of BFS?" ranked a Bellman-Ford
chunk *above* the chunk that actually states BFS's own O(V + E) complexity.
all-MiniLM-L6-v2 places "BFS", "Bellman-Ford", and "shortest path" close
together in embedding space because they're topically related graph-
algorithm terms - exactly the failure mode dense retrieval has on technical
text full of near-synonymous jargon and acronyms. BM25 does not have this
problem for this query: it directly rewards the chunk containing the literal
token "bfs", pulling it back to the top regardless of what the embedding
model thinks is "semantically similar." Fusing the two lets the sparse
signal correct the dense signal's blind spot on exact terms, while still
getting the dense signal's advantage on queries that use different words
than the source text (paraphrases, definitions, "how do I..." questions).
"""

from __future__ import annotations

from hybridrag.retrieval.dense import DenseIndex
from hybridrag.retrieval.fusion import reciprocal_rank_fusion
from hybridrag.retrieval.sparse import BM25Index
from hybridrag.types import ScoredChunk


class HybridRetriever:
    def __init__(self, dense_index: DenseIndex, sparse_index: BM25Index, rrf_k: int = 60) -> None:
        self.dense_index = dense_index
        self.sparse_index = sparse_index
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
    ) -> list[ScoredChunk]:
        """Fetch `dense_top_k` and `sparse_top_k` candidates independently,
        fuse by RRF, and return the top `top_k` fused results.

        Pulling more candidates from each retriever than the final `top_k`
        gives RRF more to work with - a chunk ranked #15 by dense but #2 by
        sparse should still surface near the top of the fusion, which it
        can't do if dense only ever handed over its top 5.
        """
        dense_results = self.dense_index.search(query, top_k=dense_top_k)
        sparse_results = self.sparse_index.search(query, top_k=sparse_top_k)
        fused = reciprocal_rank_fusion([dense_results, sparse_results], k=self.rrf_k)
        return fused[:top_k]
