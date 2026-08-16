from __future__ import annotations

from hybridrag.retrieval.reranker import CrossEncoderReranker
from hybridrag.types import Chunk, ScoredChunk


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id="doc.txt", text=text, start_char=0, end_char=len(text), chunk_index=0)


class FakeCrossEncoder:
    """Scores each (query, chunk_text) pair by a lookup table, so tests are
    deterministic without downloading a real cross-encoder model."""

    def __init__(self, scores_by_text: dict[str, float]) -> None:
        self._scores_by_text = scores_by_text

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [self._scores_by_text[text] for _, text in pairs]


class TestCrossEncoderReranker:
    def test_reorders_candidates_by_cross_encoder_score(self):
        # Deliberately give the *first* (retrieval-ranked) candidate the
        # *lowest* cross-encoder score, so a passing test proves reranking
        # actually changed the order rather than just passing it through.
        fake_model = FakeCrossEncoder({"low relevance": 0.1, "high relevance": 0.9})
        reranker = CrossEncoderReranker(model=fake_model)

        candidates = [
            ScoredChunk(chunk=_chunk("c_low", "low relevance"), score=0.8),
            ScoredChunk(chunk=_chunk("c_high", "high relevance"), score=0.2),
        ]
        results = reranker.rerank("query", candidates, top_k=2)

        assert [r.chunk.chunk_id for r in results] == ["c_high", "c_low"]
        assert results[0].score == 0.9
        assert results[1].score == 0.1

    def test_respects_top_k(self):
        fake_model = FakeCrossEncoder({"a": 0.9, "b": 0.5, "c": 0.1})
        reranker = CrossEncoderReranker(model=fake_model)

        candidates = [
            ScoredChunk(chunk=_chunk("c_a", "a"), score=0.0),
            ScoredChunk(chunk=_chunk("c_b", "b"), score=0.0),
            ScoredChunk(chunk=_chunk("c_c", "c"), score=0.0),
        ]
        results = reranker.rerank("query", candidates, top_k=1)

        assert len(results) == 1
        assert results[0].chunk.chunk_id == "c_a"

    def test_empty_candidates_returns_empty_list(self):
        reranker = CrossEncoderReranker(model=FakeCrossEncoder({}))
        assert reranker.rerank("query", [], top_k=5) == []

    def test_records_last_latency_seconds(self):
        fake_model = FakeCrossEncoder({"x": 0.5})
        reranker = CrossEncoderReranker(model=fake_model)

        assert reranker.last_latency_seconds == 0.0
        reranker.rerank("query", [ScoredChunk(chunk=_chunk("c_x", "x"), score=0.0)], top_k=1)
        assert reranker.last_latency_seconds >= 0.0

    def test_top_k_larger_than_candidate_count_returns_all_candidates(self):
        fake_model = FakeCrossEncoder({"a": 0.9, "b": 0.1})
        reranker = CrossEncoderReranker(model=fake_model)

        candidates = [
            ScoredChunk(chunk=_chunk("c_a", "a"), score=0.0),
            ScoredChunk(chunk=_chunk("c_b", "b"), score=0.0),
        ]
        results = reranker.rerank("query", candidates, top_k=10)
        assert len(results) == 2
