"""Runs the labeled eval set through a RagPipeline and aggregates results.

Retrieval metrics are always computed (cheap, deterministic, no API calls).
Generation and judge scoring are opt-in per run() call, since they cost real
API tokens/latency - a quick "did I break retrieval" check shouldn't have to
pay for 43 Claude generation calls plus 43 judge calls every time.
"""

from __future__ import annotations

import logging
import statistics
import time

from hybridrag.eval.judge import LLMJudge
from hybridrag.eval.retrieval_metrics import precision_at_k, recall_at_k, reciprocal_rank
from hybridrag.eval.types import EvalExample, EvalReport, QueryResult, RetrievalScore
from hybridrag.pipeline import RagPipeline

logger = logging.getLogger(__name__)

DEFAULT_K_VALUES = [1, 3, 5]


class EvalHarness:
    def __init__(
        self,
        pipeline: RagPipeline,
        judge: LLMJudge | None = None,
        k_values: list[int] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.judge = judge
        self.k_values = k_values or DEFAULT_K_VALUES

    def run(
        self,
        examples: list[EvalExample],
        config_name: str,
        run_generation: bool = False,
        run_judge: bool = False,
    ) -> EvalReport:
        if run_judge and self.judge is None:
            raise ValueError("run_judge=True requires an LLMJudge to be passed to EvalHarness")
        if run_judge and not run_generation:
            raise ValueError("run_judge=True requires run_generation=True (judge scores answers)")

        max_k = max(self.k_values)
        query_results: list[QueryResult] = []

        for i, example in enumerate(examples, start=1):
            logger.info("[%d/%d] %s", i, len(examples), example.query)
            relevant_set = set(example.relevant_chunk_ids)

            t0 = time.perf_counter()
            retrieved = self.pipeline.retrieve(example.query, top_k=max_k)
            retrieval_latency = time.perf_counter() - t0
            retrieved_ids = [sc.chunk.chunk_id for sc in retrieved]

            retrieval_score = RetrievalScore(
                query=example.query,
                retrieved_chunk_ids=retrieved_ids,
                relevant_chunk_ids=example.relevant_chunk_ids,
                precision_at_k={k: precision_at_k(retrieved_ids, relevant_set, k) for k in self.k_values},
                recall_at_k={k: recall_at_k(retrieved_ids, relevant_set, k) for k in self.k_values},
                reciprocal_rank=reciprocal_rank(retrieved_ids, relevant_set),
                retrieval_latency_seconds=retrieval_latency,
            )

            generation_result = None
            judge_score = None
            if run_generation:
                generation_result = self.pipeline.generate_from_chunks(example.query, retrieved)
                if run_judge:
                    judge_score = self.judge.score(example.query, generation_result, retrieved)

            query_results.append(
                QueryResult(retrieval=retrieval_score, generation=generation_result, judge=judge_score)
            )

        return _aggregate(config_name, query_results, self.k_values)


def _aggregate(config_name: str, query_results: list[QueryResult], k_values: list[int]) -> EvalReport:
    report = EvalReport(config_name=config_name, per_query=query_results)
    if not query_results:
        return report

    report.mean_precision_at_k = {
        k: statistics.mean(qr.retrieval.precision_at_k[k] for qr in query_results) for k in k_values
    }
    report.mean_recall_at_k = {
        k: statistics.mean(qr.retrieval.recall_at_k[k] for qr in query_results) for k in k_values
    }
    report.mean_reciprocal_rank = statistics.mean(qr.retrieval.reciprocal_rank for qr in query_results)
    report.mean_retrieval_latency_seconds = statistics.mean(
        qr.retrieval.retrieval_latency_seconds for qr in query_results
    )

    generations = [qr.generation for qr in query_results if qr.generation is not None]
    if generations:
        report.mean_generation_latency_seconds = statistics.mean(g.latency_seconds for g in generations)
        report.total_generation_input_tokens = sum(g.input_tokens for g in generations)
        report.total_generation_output_tokens = sum(g.output_tokens for g in generations)

    judged = [qr.judge for qr in query_results if qr.judge is not None]
    if judged:
        report.mean_faithfulness_score = statistics.mean(j.faithfulness_score for j in judged)
        report.mean_relevance_score = statistics.mean(j.relevance_score for j in judged)
        report.mean_judge_latency_seconds = statistics.mean(j.judge_latency_seconds for j in judged)
        report.total_judge_input_tokens = sum(j.judge_input_tokens for j in judged)
        report.total_judge_output_tokens = sum(j.judge_output_tokens for j in judged)

    return report
