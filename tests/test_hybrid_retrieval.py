from __future__ import annotations

from hybridrag.retrieval.hybrid import HybridRetriever
from hybridrag.types import Chunk, ScoredChunk


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id="doc.txt", text=chunk_id, start_char=0, end_char=0, chunk_index=0)


class FakeIndex:
    """Returns a fixed, pre-baked ranking regardless of the query - lets the
    test control exactly what each retriever "found" without a real model."""

    def __init__(self, ranked_chunk_ids: list[str]) -> None:
        self._ranking = [ScoredChunk(chunk=_chunk(cid), score=1.0) for cid in ranked_chunk_ids]

    def search(self, query: str, top_k: int = 20) -> list[ScoredChunk]:
        return self._ranking[:top_k]


class TestHybridRetriever:
    def test_fuses_dense_and_sparse_rankings(self):
        dense = FakeIndex(["a", "b", "c"])
        sparse = FakeIndex(["b", "a", "c"])
        retriever = HybridRetriever(dense, sparse, rrf_k=60)

        fused = retriever.retrieve("query", top_k=3)
        assert {sc.chunk.chunk_id for sc in fused} == {"a", "b", "c"}

    def test_sparse_only_hit_still_surfaces_in_fused_results(self):
        # Concrete regression case for the motivation in hybrid.py: a chunk
        # dense retrieval ranks poorly (or misses) but sparse ranks #1
        # should still make it into the fused top_k.
        dense = FakeIndex(["unrelated_1", "unrelated_2", "unrelated_3"])
        sparse = FakeIndex(["exact_match", "unrelated_1", "unrelated_2"])
        retriever = HybridRetriever(dense, sparse, rrf_k=60)

        fused = retriever.retrieve("BFS", top_k=3, dense_top_k=3, sparse_top_k=3)
        fused_ids = [sc.chunk.chunk_id for sc in fused]
        assert "exact_match" in fused_ids

    def test_respects_top_k_after_fusion(self):
        dense = FakeIndex(["a", "b", "c", "d", "e"])
        sparse = FakeIndex(["e", "d", "c", "b", "a"])
        retriever = HybridRetriever(dense, sparse, rrf_k=60)

        fused = retriever.retrieve("query", top_k=2, dense_top_k=5, sparse_top_k=5)
        assert len(fused) == 2

    def test_dense_top_k_and_sparse_top_k_limit_candidates_before_fusion(self):
        dense = FakeIndex(["a", "b", "c"])
        sparse = FakeIndex(["z"])  # only 1 candidate available from sparse
        retriever = HybridRetriever(dense, sparse, rrf_k=60)

        fused = retriever.retrieve("query", top_k=10, dense_top_k=1, sparse_top_k=1)
        # Only "a" (dense_top_k=1) and "z" (sparse's only candidate) should appear.
        assert {sc.chunk.chunk_id for sc in fused} == {"a", "z"}
