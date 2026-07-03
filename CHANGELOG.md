# Changelog

All notable changes to this project are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
**Patch version bumps** (`0.1.1`, `0.1.2`, …) happen when each feature branch merges to `dev` — see [docs/VERSIONING.md](docs/VERSIONING.md).

## [Unreleased]

## [0.2.1] - 2026-07-03

### Added — Task-specific truncation limits + CI + RUNBOOK (Phase 2b Priorities 3, 5, 6)

- **`TruncationInfo` Pydantic DTO** in `src/core/models.py` — records `chars_original`, `chars_sent`, `was_truncated`, and `limit_applied` per document.
- **`DEFAULT_TASK_TRUNCATION_LIMITS`** in `src/core/models.py` — hard-coded fallback limits per task (`translation: 2000`, `summarisation: 8000`, `full: 4000`).
- **`max_document_length_per_task`** block in `configs/config.yaml` — per-task truncation limits that override the global `max_document_length` at inference time.
- **`PipelineResult.truncation`** optional field — populated by the orchestrator on every run; original document text is always preserved in `PipelineResult.document.raw_text`.
- **`PipelineOrchestrator._get_task_truncation_limit()`** — resolves the effective limit with the priority: per-task config → global config → hard-coded default.
- **`PipelineOrchestrator._truncate_document()`** — produces a truncated document copy + `TruncationInfo`; does not mutate the original.
- **Manifest `config_snapshot` enrichment** — `truncation_limit_applied` and `task` keys added to every manifest's `config_snapshot` so the exact limit is auditable after the fact.
- **`tests/test_truncation.py`** — 18 tests covering `TruncationInfo`, limit resolution (per-task / global / default fallback), document truncation, `PipelineResult` population, and manifest snapshot.
- **`.github/workflows/test.yml`** — GitHub Actions CI pipeline running `pytest` on every push and pull request to `main` and `dev`. Caches the `uv` environment; matrix-ready for future Python version pinning.
- **`RUNBOOK.md`** — Manifest-first workflow guide: production run-and-evaluate, re-evaluate from an existing manifest, hash mismatch handling, and golden rebenchmark checklist.
- **`docs/learning/ci_pipeline_tutorial.md`** — Hands-on CI pipeline tutorial explaining GitHub Actions structure, cache strategy, pytest integration, and lessons learned from this project.

### Changed

- `PipelineOrchestrator._process_single()` now applies per-task truncation before calling any processor method. The untruncated original is stored in `PipelineResult.document`.

## [0.2.0] - 2026-07-03

### Added — Run manifest & provenance (Phase 2b Priority 1)

- **`src/core/provenance.py`** — `file_hash`, `dataset_hash`, `ground_truth_hash`, `config_hash`, `config_snapshot` utilities. Ground truth hash is computed over only the exact `doc_ids` in the run, so swapping or modifying the reference file is detected.
- **`RunManifest` Pydantic DTO** in `src/core/models.py` — full provenance record written alongside every results JSON: `run_id`, `app_version`, `task`, `model_key`, `model_id`, dataset/ground-truth/config hashes, `doc_ids`, `sample_indices`, `results_path`, `created_at`.
- **`TASK_GROUND_TRUTH_KEY` dict** in `src/core/models.py` — shared mapping of task → ground truth metadata key (used by both orchestrator and evaluator without circular import).
- **Orchestrator manifest writing** — every `PipelineOrchestrator` run now writes `{results_stem}.manifest.json` alongside the results JSON. `PipelineOrchestrator.__init__` accepts optional `model_key`, `dataset_path`, and `ground_truth_path` for full hash capture.
- **`--dataset <catalog_key>`** CLI flag on orchestrator (mutually exclusive with `--input`). Enables full manifest hash tracking from a named dataset.
- **`--evaluate`** CLI flag on orchestrator — chains evaluation immediately after inference using the manifest (no manual path bookkeeping).
- **`Evaluator.run_on_manifest(manifest_path)`** — the recommended production evaluation path. Verifies ground truth hash, audits `doc_ids`, then runs scoring. Raises `RuntimeError` on hash mismatch and `FileNotFoundError` if the ground truth file is missing.
- **`verify_manifest_ground_truth(manifest)`** and **`audit_doc_ids(results, manifest)`** — standalone helpers for hash verification and doc_id cross-checking.
- **`EvaluationReport.run_id` and `EvaluationReport.manifest_path`** — optional fields linking every report back to the manifest it was produced from.
- **`--run <manifest.json>`** CLI flag on evaluator (replaces implicit latest resolution). Requires `--results` or `--latest` to be explicit; `--latest` is now a named dev-only flag that prints a warning.
- **`tests/test_provenance.py`** — 15 tests for hash stability, subset independence, order independence, known values.
- **`tests/test_manifest.py`** — 13 tests for `RunManifest` round-trip, orchestrator manifest writing, hash verification (pass/fail/empty), doc_id audit, and end-to-end `run_on_manifest`.

### Changed

- `EvaluationReport` gains optional `run_id` and `manifest_path` fields (fully backward-compatible).
- Evaluator CLI: `--results` and `--latest` are now explicit modes, not a silent default. `--latest` prints a dev warning. `--run` is the new production default.
- Orchestrator `_save_results` returns only `Path` (unchanged); manifest writing is a side effect.

## [0.1.7] - 2026-07-03

### Fixed

- Pydantic `model_*` field warnings via shared `AppModel` base DTO
- Deprecated `datetime.utcnow()` replaced with centralized UTC helpers in `src/core/time.py`

### Added

- `tests/test_time.py` for UTC timestamp utilities

## [0.1.6] - 2026-07-02

`feature/integration` — Phase 2 complete.

### Added

- Import smoke tests for pricing, retry, providers, and benchmark modules
- End-to-end benchmark test with two mocked providers and JSON/CSV output
- `anthropic` and `openai` in `requirements.txt`

## [0.1.5] - 2026-07-02

`feature/benchmark-runner` — multi-model benchmark platform.

### Added

- `BenchmarkRunner` with fair same-document multi-model comparison
- Token, cost, and latency aggregation into `ModelBenchmarkResult`
- JSON and CSV benchmark report export
- Console comparison table
- CLI: `python -m src.evaluations.benchmark --task ... --models ...`
- `Evaluator.run_on_results()` for in-memory evaluation

## [0.1.4] - 2026-07-02

`feature/factory-cli` — processor factory tests.

### Added

- `tests/test_factory.py` — dispatch tests for Gemini, Claude, and OpenAI-compatible providers

## [0.1.3] - 2026-07-02

`feature/providers` — multi-provider adapters.

### Added

- Claude processor (Anthropic Messages API)
- OpenAI-compatible processor (OpenAI + OpenRouter)
- Gemini token usage capture and retry-wrapped API calls
- Parametrized processor contract tests
- `anthropic` and `openai` dependencies

### Changed

- `build_processor()` dispatches to Claude and OpenAI-compatible providers

## [0.1.2] - 2026-07-02

`feature/pricing-retry-models` — token cost and resilience.

### Added

- `TokenUsage` and `calculate_cost()` in `src/core/pricing.py`
- `retry_with_backoff()` decorator (tenacity)
- `token_usage` and `cost_usd` on step result DTOs
- `ModelBenchmarkResult` and `BenchmarkReport` DTOs

## [0.1.1] - 2026-07-02

`feature/config-catalog` — config-driven model catalog.

### Added

- `models.catalog` in `config.yaml` with `provider_type`, pricing, and API key env vars
- `validate_model_catalog()` and `validate_model_key()`
- `get_processed_path()` derived from `sample_size`
- `OPENROUTER_API_KEY` in `.env.example`

### Changed

- `build_processor()` reads catalog instead of hardcoded model names
- CLI `--model` validates against catalog keys

## [0.1.0] - 2026-05-17

Phase 1 — single-model Gemini pipeline with evaluation metrics.

### Added

- `BaseDocumentProcessor` abstract interface
- `GeminiProcessor` for extraction, translation, summarisation
- EuroParl and CNN/DailyMail data loaders
- BLEU, ROUGE-L, and BERTScore evaluation
- Pydantic DTOs for pipeline and evaluation results
- Centralised prompts in `configs/prompts.yaml`

[Unreleased]: https://github.com/BigEyesDev/LLMevallab/compare/v0.1.7...dev
[0.1.7]: https://github.com/BigEyesDev/LLMevallab/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/BigEyesDev/LLMevallab/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/BigEyesDev/LLMevallab/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/BigEyesDev/LLMevallab/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/BigEyesDev/LLMevallab/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/BigEyesDev/LLMevallab/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/BigEyesDev/LLMevallab/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/BigEyesDev/LLMevallab/releases/tag/v0.1.0
