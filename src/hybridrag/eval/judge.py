"""Claude-based LLM-as-judge: scores a generated answer for faithfulness
(is every claim grounded in the retrieved context?) and answer relevance
(does the answer actually address the query?), each 1-5.

Design decisions worth being able to defend:

Same model as generation, not a separate/stronger judge model. The
alternative - e.g. a larger model as judge - reduces same-model
self-preference bias (a model tends to rate its own outputs slightly more
favorably than an independent judge would), at the cost of a second model
to justify and extra spend. For a small, hand-labeled eval set that gets
spot-checked against the `reasoning` field this judge returns, that bias is
a known, disclosed limitation rather than a hidden one - not eliminating it
is a considered tradeoff, not an oversight.

One combined judge call per query, not two. Faithfulness and relevance are
conceptually separate rubrics, but they're evaluated against the exact same
context (query, answer, retrieved chunks), so splitting them into two API
calls would double latency and token cost for no accuracy benefit.

Structured output via forced tool use, not free-text-then-regex. Asking the
model to emit "Score: 4" in prose and parsing it with a regex is a real
failure mode (formatting drift breaks the parser silently). Forcing a tool
call with a JSON schema makes malformed output a hard API-level error
instead of a silently-wrong parse.
"""

from __future__ import annotations

import os
import time

import anthropic

from hybridrag.eval.types import JudgeScore
from hybridrag.types import GenerationResult, ScoredChunk

JUDGE_SYSTEM_PROMPT = """You are a strict, impartial evaluator of a RAG (retrieval-augmented \
generation) system's output. You will be given a user query, the context chunks that were \
retrieved and given to the generator, and the generator's answer (with inline [chunk_id] \
citations). Score the answer on two independent dimensions and call the submit_scores tool \
with your scores.

FAITHFULNESS (1-5): Is every factual claim in the answer actually supported by the cited \
chunk's text? An answer that cites a chunk which doesn't actually contain the claimed \
information is unfaithful, even if the claim happens to be true in general.
  5 = every claim is directly and correctly supported by its cited chunk.
  3 = mostly grounded, but at least one claim is a stretch or loosely supported.
  1 = contains claims not supported by the cited chunks, or cites a chunk that doesn't \
support the claim, or fabricates a citation.

ANSWER RELEVANCE (1-5): Does the answer actually address the user's query?
  5 = directly and completely answers what was asked.
  3 = partially answers, or answers a related but different question.
  1 = does not address the query at all.

Be strict: a fluent, confident-sounding answer is not automatically faithful or relevant."""

SUBMIT_SCORES_TOOL = {
    "name": "submit_scores",
    "description": "Submit faithfulness and relevance scores for the answer being evaluated.",
    "input_schema": {
        "type": "object",
        "properties": {
            "faithfulness_score": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "1-5 faithfulness score.",
            },
            "faithfulness_reasoning": {
                "type": "string",
                "description": "1-2 sentences justifying the faithfulness score, citing specifics.",
            },
            "relevance_score": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "1-5 answer relevance score.",
            },
            "relevance_reasoning": {
                "type": "string",
                "description": "1-2 sentences justifying the relevance score.",
            },
        },
        "required": [
            "faithfulness_score",
            "faithfulness_reasoning",
            "relevance_score",
            "relevance_reasoning",
        ],
    },
}


def _format_context(chunks: list[ScoredChunk]) -> str:
    return "\n\n".join(f"[chunk_id: {sc.chunk.chunk_id}]\n{sc.chunk.text}" for sc in chunks)


class LLMJudge:
    def __init__(
        self,
        model: str = "claude-sonnet-5",
        max_tokens: int = 512,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it before running the judge: "
                "export ANTHROPIC_API_KEY=sk-ant-..."
            )
        self._client = anthropic.Anthropic(api_key=resolved_key)

    def score(
        self,
        query: str,
        generation: GenerationResult,
        retrieved_chunks: list[ScoredChunk],
    ) -> JudgeScore:
        context = _format_context(retrieved_chunks)
        user_message = (
            f"Query: {query}\n\n"
            f"Retrieved context:\n\n{context}\n\n"
            f"Generated answer:\n{generation.answer}"
        )

        start = time.perf_counter()
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=JUDGE_SYSTEM_PROMPT,
            tools=[SUBMIT_SCORES_TOOL],
            tool_choice={"type": "tool", "name": "submit_scores"},
            messages=[{"role": "user", "content": user_message}],
        )
        latency = time.perf_counter() - start

        tool_use_block = next(b for b in response.content if b.type == "tool_use")
        scores = tool_use_block.input

        return JudgeScore(
            query=query,
            faithfulness_score=scores["faithfulness_score"],
            faithfulness_reasoning=scores["faithfulness_reasoning"],
            relevance_score=scores["relevance_score"],
            relevance_reasoning=scores["relevance_reasoning"],
            judge_latency_seconds=latency,
            judge_input_tokens=response.usage.input_tokens,
            judge_output_tokens=response.usage.output_tokens,
        )
