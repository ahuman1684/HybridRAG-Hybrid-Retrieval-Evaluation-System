"""Claude-backed generation stage.

The prompt forces per-claim citations in an exact `[chunk_id]` format for two
reasons beyond "nice to have": (1) it's what makes faithfulness measurable in
Phase 4 - the judge can check each cited chunk actually supports its claim,
rather than eyeballing the whole answer against the whole context; (2) a
citation that doesn't match any chunk_id we actually retrieved is itself a
faithfulness signal (the model citing a source it wasn't given), so we keep
the raw citation strings rather than silently dropping unmatched ones.
"""

from __future__ import annotations

import os
import re
import time

import anthropic

from hybridrag.types import Citation, GenerationResult, ScoredChunk

_CITATION_RE = re.compile(r"\[([^\[\]]+::chunk_\d+)\]")

SYSTEM_PROMPT = """You are a precise research assistant answering questions using only \
the provided context chunks.

Rules:
1. Use ONLY information present in the context chunks below. Do not use outside knowledge.
2. Every factual claim you make MUST be immediately followed by a citation in the exact \
format [chunk_id], copying the chunk_id verbatim from the context (e.g. [notes.md::chunk_2]).
3. If a sentence draws on multiple chunks, cite all of them: [chunk_a] [chunk_b].
4. If the context does not contain enough information to answer, say so explicitly instead \
of guessing or using outside knowledge.
5. Never invent a chunk_id that was not given to you in the context."""


def _format_context(chunks: list[ScoredChunk]) -> str:
    blocks = []
    for sc in chunks:
        blocks.append(f"[chunk_id: {sc.chunk.chunk_id}]\n{sc.chunk.text}")
    return "\n\n".join(blocks)


def _extract_citations(answer: str) -> list[Citation]:
    """Pull (chunk_id, claim) pairs out of the answer text.

    `claim` is the answer text between the previous citation (or the start of
    the answer) and this one - a cheap approximation of "the sentence this
    citation is attached to" that avoids a second LLM call just to segment
    claims.
    """
    citations = []
    cursor = 0
    for match in _CITATION_RE.finditer(answer):
        claim = answer[cursor : match.start()].strip(" \n[]")
        # Strip any trailing citation markers left over from a multi-citation claim.
        claim = _CITATION_RE.sub("", claim).strip()
        if claim:
            citations.append(Citation(chunk_id=match.group(1), claim=claim))
        cursor = match.end()
    return citations


class Generator:
    def __init__(
        self,
        model: str = "claude-sonnet-5",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it before running generation: "
                "export ANTHROPIC_API_KEY=sk-ant-..."
            )
        self._client = anthropic.Anthropic(api_key=resolved_key)

    def generate(self, query: str, chunks: list[ScoredChunk]) -> GenerationResult:
        context = _format_context(chunks)
        user_message = f"Context:\n\n{context}\n\nQuestion: {query}"

        start = time.perf_counter()
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        latency = time.perf_counter() - start

        answer = "".join(block.text for block in response.content if block.type == "text")
        citations = _extract_citations(answer)

        return GenerationResult(
            answer=answer,
            citations=citations,
            raw_response=answer,
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_seconds=latency,
        )
