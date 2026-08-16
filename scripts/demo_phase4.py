"""Phase 4 demo: run the hand-labeled eval set through the current pipeline
config and print real retrieval metrics (P@k, R@k, MRR). If ANTHROPIC_API_KEY
is set, also runs generation + LLM-as-judge scoring for faithfulness and
answer relevance, with separate latency/token accounting for the production
pipeline vs. the judge's own (evaluation-only) cost.

Run with:
    python scripts/demo_phase4.py
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hybridrag.config import PipelineConfig  # noqa: E402
from hybridrag.eval.harness import EvalHarness  # noqa: E402
from hybridrag.eval.judge import LLMJudge  # noqa: E402
from hybridrag.eval.labels import load_eval_examples  # noqa: E402
from hybridrag.eval.types import EvalReport  # noqa: E402
from hybridrag.pipeline import RagPipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
for _noisy in ("httpx", "huggingface_hub", "sentence_transformers"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "sample_corpus"
LABELS_PATH = ROOT / "data" / "eval" / "labels.json"


def print_report(report: EvalReport) -> None:
    print("=" * 70)
    print(f"CONFIG: {report.config_name}  ({len(report.per_query)} queries)")
    print("-" * 70)
    for k in sorted(report.mean_precision_at_k):
        print(f"  P@{k}: {report.mean_precision_at_k[k]:.3f}   R@{k}: {report.mean_recall_at_k[k]:.3f}")
    print(f"  MRR: {report.mean_reciprocal_rank:.3f}")
    print(f"  Mean retrieval latency: {report.mean_retrieval_latency_seconds * 1000:.1f}ms")

    if report.mean_generation_latency_seconds is not None:
        print(f"  Mean generation latency: {report.mean_generation_latency_seconds:.2f}s")
        print(
            f"  Generation tokens (total): {report.total_generation_input_tokens} in / "
            f"{report.total_generation_output_tokens} out"
        )
    if report.mean_faithfulness_score is not None:
        print(f"  Mean faithfulness (1-5): {report.mean_faithfulness_score:.2f}")
        print(f"  Mean answer relevance (1-5): {report.mean_relevance_score:.2f}")
        print(f"  Mean judge latency: {report.mean_judge_latency_seconds:.2f}s  (eval-time only cost)")
        print(
            f"  Judge tokens (total): {report.total_judge_input_tokens} in / "
            f"{report.total_judge_output_tokens} out"
        )

    # Show the worst 3 queries by MRR so failures are visible, not just averages.
    worst = sorted(report.per_query, key=lambda qr: qr.retrieval.reciprocal_rank)[:3]
    if worst and worst[0].retrieval.reciprocal_rank < 1.0:
        print("\n  Lowest-scoring queries (by reciprocal rank):")
        for qr in worst:
            if qr.retrieval.reciprocal_rank >= 1.0:
                continue
            print(f"    RR={qr.retrieval.reciprocal_rank:.2f}  \"{qr.retrieval.query}\"")
            print(f"      expected: {qr.retrieval.relevant_chunk_ids}")
            print(f"      got top-3: {qr.retrieval.retrieved_chunk_ids[:3]}")
    print()


def main() -> None:
    examples = load_eval_examples(LABELS_PATH)
    print(f"Loaded {len(examples)} labeled examples from {LABELS_PATH}")

    config = PipelineConfig()
    config.chunking.strategy = "fixed"

    pipeline = RagPipeline(config)
    t0 = time.perf_counter()
    pipeline.ingest(str(CORPUS_DIR))
    print(f"Ingested corpus in {time.perf_counter() - t0:.2f}s\n")

    has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    judge = LLMJudge() if has_api_key else None
    harness = EvalHarness(pipeline, judge=judge)

    if not has_api_key:
        print("ANTHROPIC_API_KEY not set - running retrieval metrics only.\n")

    report = harness.run(
        examples,
        config_name="hybrid+rerank (current default config)",
        run_generation=has_api_key,
        run_judge=has_api_key,
    )
    print_report(report)


if __name__ == "__main__":
    main()
