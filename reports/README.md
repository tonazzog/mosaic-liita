# Evaluation reports

Reports in this folder are the ones used in the thesis: canonical gold
(`data/test_dataset.json`, hypolemmas included), unified F1 evaluator, run on
3–4 November 2026.

| File | Avg F1 | Notes |
| --- | --- | --- |
| `f1_report_mosaic_agentic_mistral_mistral-large-4-0.json` | 0.860 | Agentic mode, `mistral-large-4-0`, after the block fixes (affix hyphens, hypolemmas, per-item average, DISTINCT/definitions) |
| `f1_report_mosaic_deterministic_canonical.json` | 0.484 | Deterministic planner (no LLM), after the same fixes |

`.calls.jsonl` files have one line per LLM call with token usage; `.log` files are
the console output.

## archive/

Not used for the thesis results.

- `superseded/` – same setup, earlier code (`..._prefix`): agentic 0.737,
  deterministic 0.428 (before the block fixes).
- `2026-03/` – March 2026 runs (Claude Sonnet 4.6, Mistral Large 3 via Ollama,
  deterministic) on the older gold and evaluator; not comparable.
- `claudecode/` – runs through the Claude Code CLI (October 2026), made before the
  canonical gold and the current code; `opus_subset` covers only 58 questions.
  To be redone if Claude results are wanted.
