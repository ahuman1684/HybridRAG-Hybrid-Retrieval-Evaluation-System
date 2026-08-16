# hybridrag

A hybrid (dense + sparse) retrieval-augmented generation pipeline with a
rigorous evaluation and ablation harness, built to be defensible in a
technical interview - every architectural choice is measured, not assumed.

> Status: Phase 1 (baseline dense-only pipeline) complete. Architecture
> diagram, setup instructions, and the ablation results table land in
> Phase 6 once the full pipeline and eval harness are built.

## Quickstart (Phase 1)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

export ANTHROPIC_API_KEY=sk-ant-...  # optional, only needed for generation
python scripts/demo_phase1.py
pytest
```
