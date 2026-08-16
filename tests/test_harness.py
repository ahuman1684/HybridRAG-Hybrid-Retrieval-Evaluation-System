from __future__ import annotations

import pytest

from hybridrag.eval.harness import EvalHarness
from hybridrag.eval.types import EvalExample, JudgeScore
from hybridrag.types import Chunk, GenerationResult, ScoredChunk


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id="doc.txt", text=chunk_id, start_char=0, end_char=0, chunk_index=0)


class FakePipeline:
    """Duck-typed stand-in for RagPipeline - avoids loading a real embedding
    model just to test aggregation logic. Returns a fixed ranking per query,
    looked up by query string, so tests can control retrieval precisely."""

    def __init__(self, rankings_by_query: dict[str, list[str]]) -> None:
        self._rankings_by_query = rankings_by_query
        self.generate_calls: list[tuple[str, list[ScoredChunk]]] = []

    def retrieve(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        ids = self._rankings_by_query[query][:top_k]
        return [ScoredChunk(chunk=_chunk(cid), score=1.0) for cid in ids]

    def generate_from_chunks(self, query: str, chunks: list[ScoredChunk]) -> GenerationResult:
        self.generate_calls.append((query, chunks))
        return GenerationResult(
            answer=f"answer for {query}",
            citations=[],
            raw_response="",
            model="fake-model",
            input_tokens=100,
            output_tokens=20,
            latency_seconds=0.5,
        )


class FakeJudge:
    def __init__(self) -> None:
        self.score_calls: list[str] = []

    def score(self, query, generation, retrieved_chunks) -> JudgeScore:
        self.score_calls.append(query)
        return JudgeScore(
            query=query,
            faithfulness_score=4,
            faithfulness_reasoning="looks fine",
            relevance_score=5,
            relevance_reasoning="on topic",
            judge_latency_seconds=0.2,
            judge_input_tokens=50,
            judge_output_tokens=10,
        )


class TestEvalHarnessRetrievalOnly:
    def test_computes_retrieval_metrics_without_generation_or_judge(self):
        pipeline = FakePipeline({"q1": ["a", "b", "c"]})
        harness = EvalHarness(pipeline, k_values=[1, 3])
        examples = [EvalExample(query="q1", relevant_chunk_ids=["a"])]

        report = harness.run(examples, config_name="dense_only")

        assert report.mean_precision_at_k[1] == 1.0
        assert report.mean_reciprocal_rank == 1.0
        assert report.mean_generation_latency_seconds is None
        assert report.mean_faithfulness_score is None
        assert len(pipeline.generate_calls) == 0

    def test_run_judge_without_generation_raises(self):
        pipeline = FakePipeline({"q1": ["a"]})
        harness = EvalHarness(pipeline, judge=FakeJudge())
        with pytest.raises(ValueError):
            harness.run([EvalExample(query="q1", relevant_chunk_ids=["a"])], "cfg", run_judge=True)

    def test_run_judge_without_judge_instance_raises(self):
        pipeline = FakePipeline({"q1": ["a"]})
        harness = EvalHarness(pipeline)  # no judge passed
        with pytest.raises(ValueError):
            harness.run(
                [EvalExample(query="q1", relevant_chunk_ids=["a"])], "cfg", run_generation=True, run_judge=True
            )


class TestEvalHarnessWithGenerationAndJudge:
    def test_reuses_retrieved_chunks_for_generation_not_a_second_retrieval(self):
        pipeline = FakePipeline({"q1": ["a", "b"]})
        harness = EvalHarness(pipeline, judge=FakeJudge(), k_values=[1, 2])
        examples = [EvalExample(query="q1", relevant_chunk_ids=["a"])]

        harness.run(examples, "cfg", run_generation=True, run_judge=True)

        assert len(pipeline.generate_calls) == 1
        _, chunks_passed = pipeline.generate_calls[0]
        assert [sc.chunk.chunk_id for sc in chunks_passed] == ["a", "b"]

    def test_aggregates_generation_and_judge_metrics_separately(self):
        pipeline = FakePipeline({"q1": ["a"], "q2": ["a"]})
        judge = FakeJudge()
        harness = EvalHarness(pipeline, judge=judge, k_values=[1])
        examples = [
            EvalExample(query="q1", relevant_chunk_ids=["a"]),
            EvalExample(query="q2", relevant_chunk_ids=["a"]),
        ]

        report = harness.run(examples, "cfg", run_generation=True, run_judge=True)

        assert report.mean_faithfulness_score == 4.0
        assert report.mean_relevance_score == 5.0
        assert report.total_generation_input_tokens == 200  # 100 * 2 queries
        assert report.total_judge_input_tokens == 100  # 50 * 2 queries - never summed with generation
        assert len(judge.score_calls) == 2

    def test_generation_without_judge_leaves_judge_fields_none(self):
        pipeline = FakePipeline({"q1": ["a"]})
        harness = EvalHarness(pipeline, k_values=[1])
        examples = [EvalExample(query="q1", relevant_chunk_ids=["a"])]

        report = harness.run(examples, "cfg", run_generation=True, run_judge=False)

        assert report.mean_generation_latency_seconds == 0.5
        assert report.mean_faithfulness_score is None
