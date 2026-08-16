from __future__ import annotations

from hybridrag.retrieval.fusion import reciprocal_rank_fusion
from hybridrag.types import Chunk, ScoredChunk


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id="doc.txt", text=chunk_id, start_char=0, end_char=0, chunk_index=0)


def _ranking(*chunk_ids: str) -> list[ScoredChunk]:
    # Raw scores here are deliberately nonsensical/out of scale with each
    # other, to prove RRF ignores them and only uses rank position.
    return [ScoredChunk(chunk=_chunk(cid), score=999.0 - i) for i, cid in enumerate(chunk_ids)]


class TestReciprocalRankFusion:
    def test_matches_hand_computed_rrf_scores(self):
        dense = _ranking("a", "b", "c")
        sparse = _ranking("b", "a", "c")

        fused = reciprocal_rank_fusion([dense, sparse], k=60)
        scores = {sc.chunk.chunk_id: sc.score for sc in fused}

        # a: rank 1 in dense (1/61), rank 2 in sparse (1/62)
        # b: rank 2 in dense (1/62), rank 1 in sparse (1/61)
        # c: rank 3 in both (1/63 + 1/63)
        assert scores["a"] == 1 / 61 + 1 / 62
        assert scores["b"] == 1 / 62 + 1 / 61
        assert scores["c"] == 1 / 63 + 1 / 63

    def test_a_and_b_tie_when_ranks_are_symmetric(self):
        dense = _ranking("a", "b")
        sparse = _ranking("b", "a")
        fused = reciprocal_rank_fusion([dense, sparse], k=60)
        assert fused[0].score == fused[1].score

    def test_chunk_present_in_both_rankings_outranks_chunk_in_only_one(self):
        dense = _ranking("a", "b", "c")
        sparse = _ranking("z", "y", "a")  # "a" also appears here, boosting it further
        fused = reciprocal_rank_fusion([dense, sparse], k=60)
        assert fused[0].chunk.chunk_id == "a"

    def test_chunk_missing_from_a_ranking_only_gets_credit_from_rankings_it_is_in(self):
        dense = _ranking("a", "b")
        sparse: list[ScoredChunk] = []  # "a" and "b" not retrieved by sparse at all
        fused = reciprocal_rank_fusion([dense, sparse], k=60)
        assert {sc.chunk.chunk_id for sc in fused} == {"a", "b"}
        assert fused[0].chunk.chunk_id == "a"

    def test_empty_rankings_yield_empty_fusion(self):
        assert reciprocal_rank_fusion([[], []], k=60) == []

    def test_single_ranking_preserves_order(self):
        ranking = _ranking("x", "y", "z")
        fused = reciprocal_rank_fusion([ranking], k=60)
        assert [sc.chunk.chunk_id for sc in fused] == ["x", "y", "z"]

    def test_lower_k_amplifies_rank_differences(self):
        dense = _ranking("a", "b")
        low_k = reciprocal_rank_fusion([dense], k=1)
        high_k = reciprocal_rank_fusion([dense], k=1000)

        low_k_gap = low_k[0].score - low_k[1].score
        high_k_gap = high_k[0].score - high_k[1].score
        assert low_k_gap > high_k_gap
