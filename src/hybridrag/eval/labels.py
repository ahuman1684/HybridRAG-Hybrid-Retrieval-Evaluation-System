"""Load the hand-constructed labeled eval set from data/eval/labels.json."""

from __future__ import annotations

import json
from pathlib import Path

from hybridrag.eval.types import EvalExample


def load_eval_examples(path: str | Path = "data/eval/labels.json") -> list[EvalExample]:
    payload = json.loads(Path(path).read_text())
    return [
        EvalExample(
            query=item["query"],
            relevant_chunk_ids=item["relevant_chunk_ids"],
            reasoning=item.get("reasoning", ""),
        )
        for item in payload["examples"]
    ]
