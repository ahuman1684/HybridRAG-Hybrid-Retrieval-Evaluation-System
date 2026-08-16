from __future__ import annotations

from types import SimpleNamespace

import pytest

from hybridrag.eval.judge import LLMJudge
from hybridrag.types import Chunk, GenerationResult, ScoredChunk


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id="doc.txt", text=text, start_char=0, end_char=len(text), chunk_index=0)


class FakeAnthropicClient:
    """Stands in for anthropic.Anthropic so tests don't make real API calls.
    Mimics just enough of the messages.create() response shape LLMJudge reads."""

    def __init__(self, tool_input: dict) -> None:
        self._tool_input = tool_input
        self.last_call_kwargs: dict | None = None
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.last_call_kwargs = kwargs
        tool_use_block = SimpleNamespace(type="tool_use", input=self._tool_input)
        return SimpleNamespace(
            content=[tool_use_block],
            usage=SimpleNamespace(input_tokens=42, output_tokens=7),
        )


def _make_judge(tool_input: dict) -> tuple[LLMJudge, FakeAnthropicClient]:
    judge = LLMJudge(api_key="fake-key-for-test")
    fake_client = FakeAnthropicClient(tool_input)
    judge._client = fake_client
    return judge, fake_client


class TestLLMJudge:
    def test_parses_scores_from_tool_use_response(self):
        judge, _ = _make_judge(
            {
                "faithfulness_score": 4,
                "faithfulness_reasoning": "mostly grounded",
                "relevance_score": 5,
                "relevance_reasoning": "directly answers",
            }
        )
        generation = GenerationResult(
            answer="BFS runs in O(V + E) time [graph.md::chunk_0].",
            citations=[],
            raw_response="",
            model="claude-sonnet-5",
            input_tokens=0,
            output_tokens=0,
            latency_seconds=0.0,
        )
        chunks = [ScoredChunk(chunk=_chunk("graph.md::chunk_0", "BFS runs in O(V + E) time."), score=1.0)]

        result = judge.score("What is BFS's complexity?", generation, chunks)

        assert result.faithfulness_score == 4
        assert result.relevance_score == 5
        assert result.judge_input_tokens == 42
        assert result.judge_output_tokens == 7

    def test_forces_the_submit_scores_tool(self):
        judge, fake_client = _make_judge(
            {
                "faithfulness_score": 3,
                "faithfulness_reasoning": "x",
                "relevance_score": 3,
                "relevance_reasoning": "y",
            }
        )
        generation = GenerationResult(
            answer="answer", citations=[], raw_response="", model="m",
            input_tokens=0, output_tokens=0, latency_seconds=0.0,
        )
        judge.score("query", generation, [ScoredChunk(chunk=_chunk("c", "text"), score=1.0)])

        assert fake_client.last_call_kwargs["tool_choice"] == {"type": "tool", "name": "submit_scores"}

    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError):
            LLMJudge()
