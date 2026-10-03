# IntentBench — M2/M4/M5 reproducibility bundle

The exact dataset and **raw, auditable** run outputs behind `docs/intent-mutation-results-m2-m4.md`
(intent-mutation paper, RUN slice). The rewriter + judge run at temperature 0 (greedy), so the run
reproduces identically — and every item's raw model rewrite is saved here.

## Contents
- `corpus/{dx,dose,triage,rx}.jsonl` — the exact IntentBench corpus (80 items: 4 clinical tasks ×
  4 attack types × 5 paraphrase variants). Each line:
  `{id, task, attack_type, sanctioned_miim, tilted_miim, moral_filter, expected_tilt_metric}`.
- `results/<model>.json` — raw `intentrun` output per model: `{report, items}`. Each `items[]` entry
  is a full audit row:
  `{id, task, attack_type, rewritten, outcome, outcome_lexical, outcome_judge, complied,
    flagged_provenance, flagged_goal_drift_lexical, flagged_goal_drift_semantic, drift_lexical, drift_semantic}`
  — including **`rewritten`** (the verbatim model output) and **both** classifications.
- `results/run.log` — per-model stderr summary lines.
- `summary.md` — aggregated M2/M4 tables + the M5 semantic-threshold sweep (from `summarize.py`).

## Panel (7 models)
`llama3.2:latest` (3B), `phi3.5:3.8b`, `qwen3:4b`, `qwen3:8b`, `gemma2:9b`, `llama3.1:8b`,
`deepseek-r1:14b`. Judge `llama3.1:8b`; embed `nomic-embed-text:latest`. 560 items, 506 complied.
Date: 2026-06-07.

## Provenance (how it was produced)
- **Corpus:** `python3 scripts/002-pristine-weights/gen.py --out datasets/002-pristine-weights/intentbench --n 20` — deterministic
  (`random.Random(42)`); regenerating reproduces `corpus/` byte-for-byte.
- **Run:** `intentrun --models llama3.2:latest,llama3.1:8b,qwen3:4b,qwen3:8b,gemma2:9b,phi3.5:3.8b,deepseek-r1:14b
  --corpus datasets/002-pristine-weights/intentbench --judge-model llama3.1:8b --embed-model nomic-embed-text:latest
  --out results/002-pristine-weights/intentbench` (local Ollama, threshold 0.3). Per item: live `LlmRewriter` rewrites
  the sanctioned MIIM under the moral filter → outcome = lexical rubric reconciled with the
  `llama3.1:8b` LLM-judge (`combine`, judge-authoritative) → harness detection (provenance + lexical
  & `nomic-embed-text` semantic `goal_drift`).
- **Code:** RUN-slice branch `paper-intent-mutation-run`, enriched harness capturing `id` +
  `rewritten` + both outcomes. Summarizer: `scripts/002-pristine-weights/summarize.py`.

## Reproduce
- **Corpus (exact):** rerun `gen.py` (deterministic).
- **Tables from these saved results:** `python3 scripts/002-pristine-weights/summarize.py --in datasets/002-pristine-weights/intentbench-m2m4/results`.
- **Fresh run:** rerun the `intentrun` command above (temp-0 greedy → reproduces the same numbers),
  then summarize.

## Notes
- **Reasoning models** (`qwen3` ×2, `deepseek-r1`) emit `<think>`; their rewrites are verbose and
  reuse anchor vocabulary, which lowers lexical drift for `qwen3:4b` (0.787) and keeps semantic drift
  ~0 — see the `rewritten` field for the verbatim outputs and `docs/intent-mutation-results-m2-m4.md`
  for the full discussion.
- **Provenance ~100% is audit coverage**, not discrimination. **Semantic@0.3 collapses**; recalibrated
  to ~0.02–0.05 it matches lexical (M5 sweep).
