# RUNBOOK — Manifest-First Workflow

> **Audience:** anyone running inference or evaluation on this project.
> **Golden rule:** every score you report must be traceable to a manifest.

---

## Why manifests?

Before Phase 2b, the evaluator silently read `latest_{task}.txt` and took the ground truth from `config.yaml`. Two things could go wrong invisibly:

1. You ran inference on Monday, came back on Thursday, and evaluated whatever happened to be the last run.
2. The ground truth file on disk might be a different version than the one used during inference.

A **RunManifest** pins the exact dataset file (hash), ground truth subset (hash), doc IDs, config snapshot, and results path to a single run ID. The evaluator re-verifies the hash before computing a single metric — wrong file → hard error, not garbage scores.

---

## Quick reference

### Production run: infer + evaluate in one command

```bash
uv run python -m src.pipeline.orchestrator \
  --task summarisation \
  --model gemini-2.5-flash \
  --dataset cnn_dailymail \
  --sample 10 \
  --evaluate
```

What happens:

```
load dataset (from config catalog)
  → apply per-task truncation (summarisation: 8000 chars)
  → run inference
  → write results JSON  +  results.manifest.json
  → evaluator reads manifest (verifies ground_truth_hash)
  → write evaluation report linked to run_id
```

Output files:

```
outputs/results/results_summarisation_gemini_2_5_flash_20260703_094208.json
outputs/results/results_summarisation_gemini_2_5_flash_20260703_094208.manifest.json
outputs/reports/report_summarisation_gemini_2_5_flash_20260703_094208.json
```

---

### Re-evaluate from an existing manifest (reproducibility check)

```bash
uv run python -m src.evaluations.evaluator \
  --run outputs/results/results_summarisation_gemini_2_5_flash_20260703_094208.manifest.json
```

- Re-hashes the ground truth for the recorded `doc_ids`.
- Raises `RuntimeError` if the hash does not match.
- Produces an identical report (same scores) if nothing changed.

---

### Explicit paths (no manifest)

Use this for legacy results files that predate manifest support, or when debugging:

```bash
uv run python -m src.evaluations.evaluator \
  --results outputs/results/results_summarisation_gemini_2_5_flash_20260703_094208.json \
  --ground-truth data/processed/cnn_dailymail/cnn_dailymail_20docs.json \
  --task summarisation
```

> No hash verification is performed in this mode.

---

### Dev-only: evaluate the latest run

```bash
uv run python -m src.evaluations.evaluator \
  --latest \
  --task summarisation
```

Prints a warning. Use `--run` in any non-throwaway workflow.

---

## Hash mismatch — what to do

You will see:

```
RuntimeError: Ground truth hash MISMATCH for run '20260703_094208_a1b2c3'.
  Manifest recorded: sha256:def456...
  Recomputed now:    sha256:aaa999...
The ground truth file 'data/processed/cnn_dailymail/cnn_dailymail_20docs.json' has changed
since inference was run. Evaluation refused to prevent misleading scores.
```

Checklist:

1. **Is the right file on disk?** Check `manifest.ground_truth_path` — ensure the file at that path has not been replaced or regenerated.
2. **Did you accidentally run `main.py` again?** Re-downloading CNN/DailyMail may shuffle the sample; doc IDs stay the same but reference summaries can differ if HF version changed.
3. **Do you still have the original file?** If yes, restore it and re-run the evaluator.
4. **Is the run truly un-reproducible?** Document it and re-run inference with `--dataset` to produce a fresh manifest-linked result.

> Never use `skip_hash_verification=True` except during debugging. It is not an acceptable workaround for production reports.

---

## Golden rebenchmark checklist

Use this when you change `max_document_length_per_task` or any other config that affects inference quality, and you want to compare old vs new scores fairly.

```
[ ] Note the old run_id from the manifest you are replacing.
[ ] Update the relevant config value (e.g. summarisation: 4000 → 8000).
[ ] Run a fresh production run with --evaluate to get a new manifest + report.
[ ] Compare old and new reports by run_id (both are persisted in outputs/reports/).
[ ] Record the change and run_ids in CHANGELOG.md under the current version.
[ ] Commit the config change and new reports together so the diff is auditable.
```

---

## Manifest anatomy

```json
{
  "run_id":             "20260703_094208_a1b2c3",
  "app_version":        "0.2.1",
  "task":               "summarisation",
  "model_key":          "gemini-2.5-flash",
  "model_id":           "gemini-2.5-flash",
  "dataset_path":       "data/processed/cnn_dailymail/cnn_dailymail_20docs.json",
  "dataset_hash":       "sha256:...",
  "doc_ids":            ["cnn_0000", "cnn_0001", "..."],
  "sample_size":        10,
  "sample_indices":     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  "ground_truth_path":  "data/processed/cnn_dailymail/cnn_dailymail_20docs.json",
  "ground_truth_hash":  "sha256:...",
  "config_hash":        "sha256:...",
  "config_snapshot": {
    "pipeline":                 { "target_language": "en", "max_document_length": 2000, ... },
    "model":                    { "model_id": "gemini-2.5-flash", "temperature": 0.1, ... },
    "truncation_limit_applied": 8000,
    "task":                     "summarisation"
  },
  "results_path":  "outputs/results/results_summarisation_gemini_2_5_flash_20260703_094208.json",
  "created_at":    "2026-07-03T09:42:08+00:00"
}
```

**Key fields:**

| Field | Purpose |
|---|---|
| `run_id` | Unique ID — links results, manifest, and report |
| `dataset_hash` | Hash of the full input file — detects dataset swaps |
| `ground_truth_hash` | Hash of reference texts for **these doc_ids only** — detects GT file changes |
| `config_hash` | Hash of pipeline + model config — detects config drift |
| `truncation_limit_applied` | Effective char limit used — auditable in every manifest |
| `doc_ids` | Exact documents processed — no ambiguity |

---

## Translation task

Replace `--task summarisation --dataset cnn_dailymail` with:

```bash
--task translation --dataset europarl
```

The effective truncation limit for translation is `2000` chars (set in `configs/config.yaml`).

---

## Running tests locally

```bash
# All offline tests (no API keys needed)
uv run pytest tests/ \
  --ignore=tests/test_gemini_processor.py \
  --ignore=tests/test_claude_processor.py \
  --ignore=tests/test_openai_compatible_processor.py \
  -v

# Specific test modules
uv run pytest tests/test_manifest.py tests/test_truncation.py -v
```

CI runs the same command on every push to `main` and `dev` via `.github/workflows/test.yml`.
