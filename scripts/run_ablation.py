"""Phase 5: the core deliverable. Runs the labeled eval set through three
pipeline configurations - (a) dense-only, (b) dense+BM25, (c)
dense+BM25+reranker - and prints a comparison table plus the per-component
deltas in accuracy vs. added latency.

Retrieval metrics always run (free, deterministic). Generation + LLM-judge
scoring run too if ANTHROPIC_API_KEY is set - across 3 configs x 43 queries
that's up to 129 generation calls + 129 judge calls, so it's opt-in and
costs real tokens; retrieval-only is the fast/free path for iterating on the
pipeline itself.

Run with:
    python scripts/run_ablation.py
    python scripts/run_ablation.py --with-generation   # also runs generation+judge (needs ANTHROPIC_API_KEY)
"""

from __future__ import annotations

import argparse
import json
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

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "sample_corpus"
LABELS_PATH = ROOT / "data" / "eval" / "labels.json"
RESULTS_DIR = ROOT / "data" / "eval" / "results"


def make_config(mode: str, reranker_enabled: bool) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.chunking.strategy = "fixed"
    cfg.retrieval.mode = mode
    cfg.reranker.enabled = reranker_enabled
    return cfg


CONFIGS: dict[str, PipelineConfig] = {
    "(a) dense-only": make_config(mode="dense", reranker_enabled=False),
    "(b) dense+BM25": make_config(mode="hybrid", reranker_enabled=False),
    "(c) dense+BM25+reranker": make_config(mode="hybrid", reranker_enabled=True),
}


def run_config(name: str, config: PipelineConfig, examples, with_generation: bool) -> EvalReport:
    print(f"Running {name}...")
    pipeline = RagPipeline(config)
    pipeline.ingest(str(CORPUS_DIR))

    judge = LLMJudge() if with_generation else None
    harness = EvalHarness(pipeline, judge=judge)
    return harness.run(examples, config_name=name, run_generation=with_generation, run_judge=with_generation)


def print_table(reports: list[EvalReport]) -> None:
    headers = ["Config", "P@1", "P@3", "R@3", "R@5", "MRR", "Retrieval ms"]
    if any(r.mean_faithfulness_score is not None for r in reports):
        headers += ["Faithfulness", "Relevance", "Gen ms", "Gen tokens (in/out)"]

    rows = []
    for r in reports:
        row = [
            r.config_name,
            f"{r.mean_precision_at_k[1]:.3f}",
            f"{r.mean_precision_at_k[3]:.3f}",
            f"{r.mean_recall_at_k[3]:.3f}",
            f"{r.mean_recall_at_k[5]:.3f}",
            f"{r.mean_reciprocal_rank:.3f}",
            f"{r.mean_retrieval_latency_seconds * 1000:.1f}",
        ]
        if r.mean_faithfulness_score is not None:
            row += [
                f"{r.mean_faithfulness_score:.2f}",
                f"{r.mean_relevance_score:.2f}",
                f"{r.mean_generation_latency_seconds * 1000:.0f}",
                f"{r.total_generation_input_tokens}/{r.total_generation_output_tokens}",
            ]
        rows.append(row)

    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    print("| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |")
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows:
        print("| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |")


def print_deltas(reports: list[EvalReport]) -> None:
    a, b, c = reports  # dense-only, dense+BM25, dense+BM25+reranker

    print("\nWhat each component earned, and what it cost:")

    mrr_gain_bm25 = b.mean_reciprocal_rank - a.mean_reciprocal_rank
    lat_cost_bm25 = (b.mean_retrieval_latency_seconds - a.mean_retrieval_latency_seconds) * 1000
    print(
        f"  + BM25 (a -> b): MRR {a.mean_reciprocal_rank:.3f} -> {b.mean_reciprocal_rank:.3f} "
        f"({mrr_gain_bm25:+.3f}), P@1 {a.mean_precision_at_k[1]:.3f} -> {b.mean_precision_at_k[1]:.3f} "
        f"({b.mean_precision_at_k[1] - a.mean_precision_at_k[1]:+.3f}), "
        f"retrieval latency {lat_cost_bm25:+.1f}ms"
    )

    mrr_gain_rerank = c.mean_reciprocal_rank - b.mean_reciprocal_rank
    lat_cost_rerank = (c.mean_retrieval_latency_seconds - b.mean_retrieval_latency_seconds) * 1000
    print(
        f"  + Reranker (b -> c): MRR {b.mean_reciprocal_rank:.3f} -> {c.mean_reciprocal_rank:.3f} "
        f"({mrr_gain_rerank:+.3f}), P@1 {b.mean_precision_at_k[1]:.3f} -> {c.mean_precision_at_k[1]:.3f} "
        f"({c.mean_precision_at_k[1] - b.mean_precision_at_k[1]:+.3f}), "
        f"retrieval latency {lat_cost_rerank:+.1f}ms"
    )

    total_mrr_gain = c.mean_reciprocal_rank - a.mean_reciprocal_rank
    total_lat_cost = (c.mean_retrieval_latency_seconds - a.mean_retrieval_latency_seconds) * 1000
    print(
        f"  Total (a -> c): MRR {total_mrr_gain:+.3f}, "
        f"P@1 {c.mean_precision_at_k[1] - a.mean_precision_at_k[1]:+.3f}, "
        f"retrieval latency {total_lat_cost:+.1f}ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-generation", action="store_true")
    args = parser.parse_args()

    with_generation = args.with_generation
    if with_generation and not os.environ.get("ANTHROPIC_API_KEY"):
        print("--with-generation requires ANTHROPIC_API_KEY to be set. Falling back to retrieval-only.")
        with_generation = False

    examples = load_eval_examples(LABELS_PATH)
    print(f"Loaded {len(examples)} labeled examples\n")

    reports = []
    t0 = time.perf_counter()
    for name, config in CONFIGS.items():
        reports.append(run_config(name, config, examples, with_generation))
    print(f"\nAblation run completed in {time.perf_counter() - t0:.1f}s\n")

    print_table(reports)
    print_deltas(reports)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "ablation_results.json"
    out_path.write_text(
        json.dumps(
            [
                {
                    "config_name": r.config_name,
                    "mean_precision_at_k": r.mean_precision_at_k,
                    "mean_recall_at_k": r.mean_recall_at_k,
                    "mean_reciprocal_rank": r.mean_reciprocal_rank,
                    "mean_retrieval_latency_seconds": r.mean_retrieval_latency_seconds,
                    "mean_faithfulness_score": r.mean_faithfulness_score,
                    "mean_relevance_score": r.mean_relevance_score,
                    "mean_generation_latency_seconds": r.mean_generation_latency_seconds,
                    "total_generation_input_tokens": r.total_generation_input_tokens,
                    "total_generation_output_tokens": r.total_generation_output_tokens,
                }
                for r in reports
            ],
            indent=2,
        )
    )
    print(f"\nSaved raw results to {out_path}")


if __name__ == "__main__":
    main()
