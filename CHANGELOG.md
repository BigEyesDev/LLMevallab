# Changelog

All notable changes to this project are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
**Patch version bumps** (`0.1.1`, `0.1.2`, …) happen when each feature branch merges to `dev` — see [docs/VERSIONING.md](docs/VERSIONING.md).

## [Unreleased]

### Added

- (pending) Integration smoke and e2e tests (`feature/integration`)

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

[Unreleased]: https://github.com/BigEyesDev/LLMevallab/compare/v0.1.5...dev
[0.1.5]: https://github.com/BigEyesDev/LLMevallab/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/BigEyesDev/LLMevallab/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/BigEyesDev/LLMevallab/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/BigEyesDev/LLMevallab/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/BigEyesDev/LLMevallab/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/BigEyesDev/LLMevallab/releases/tag/v0.1.0
