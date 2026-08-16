"""Data model for the evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass, field

from hybridrag.types import GenerationResult


@dataclass(frozen=True)
class EvalExample:
    """One hand-labeled (query, correct chunk) pair.

    `relevant_chunk_ids` is a list rather than a single id so a query whose
    answer genuinely spans multiple chunks isn't forced into a single-label
    lie - but for this corpus's chunking config, nearly every example has
    exactly one entry (see data/eval/labels.json's top-level note on how
    chunk overlap was handled when labeling).
    """

    query: str
    relevant_chunk_ids: list[str]
    reasoning: str = ""


@dataclass
class RetrievalScore:
    """Retrieval metrics for a single query."""

    query: str
    retrieved_chunk_ids: list[str]
    relevant_chunk_ids: list[str]
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    reciprocal_rank: float
    retrieval_latency_seconds: float


@dataclass
class JudgeScore:
    """LLM-as-judge output for a single query's generated answer.

    Deliberately kept separate from GenerationResult's own latency/tokens
    (see QueryResult below): the judge call is an evaluation-time cost, not
    a cost the production pipeline ever pays, so the two must never be
    summed together when reporting what the pipeline itself costs.
    """

    query: str
    faithfulness_score: int  # 1-5
    faithfulness_reasoning: str
    relevance_score: int  # 1-5
    relevance_reasoning: str
    judge_latency_seconds: float
    judge_input_tokens: int
    judge_output_tokens: int


@dataclass
class QueryResult:
    """Everything measured for one query in one pipeline configuration.

    `generation` is the pipeline's own GenerationResult (answer + its real
    latency/token cost) - what the ablation study should report as pipeline
    cost. `judge` is a separate, optional evaluation-time-only measurement.
    """

    retrieval: RetrievalScore
    generation: GenerationResult | None = None
    judge: JudgeScore | None = None


@dataclass
class EvalReport:
    """Aggregate results for a full run of the eval set against one config."""

    config_name: str
    per_query: list[QueryResult] = field(default_factory=list)

    # Aggregate retrieval metrics, keyed by k.
    mean_precision_at_k: dict[int, float] = field(default_factory=dict)
    mean_recall_at_k: dict[int, float] = field(default_factory=dict)
    mean_reciprocal_rank: float = 0.0
    mean_retrieval_latency_seconds: float = 0.0

    # Aggregate generation (production pipeline) metrics - None if
    # generation wasn't run for this report.
    mean_generation_latency_seconds: float | None = None
    total_generation_input_tokens: int | None = None
    total_generation_output_tokens: int | None = None

    # Aggregate judge (evaluation-time only) metrics - None if judging
    # wasn't run. Never added into the generation totals above.
    mean_faithfulness_score: float | None = None
    mean_relevance_score: float | None = None
    mean_judge_latency_seconds: float | None = None
    total_judge_input_tokens: int | None = None
    total_judge_output_tokens: int | None = None
