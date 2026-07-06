# Model Catalog — Test Notes & Known Quirks

This document records smoke-test results and operational notes for models in `configs/config.yaml`.
Pricing is indicative — see [OpenRouter](https://openrouter.ai/models) or provider docs for live rates.

**Last updated:** 2026-07-06 · prompt version tracked separately in `configs/prompts.yaml`.

---

## Default model

| Key | Model ID | Provider | Notes |
|---|---|---|---|
| `gemini-2.5-flash` | `gemini-2.5-flash` | Google Gemini | Default catalog entry. Fast, cheap, reliable JSON on extraction/summarisation steps. |

### Smoke-test scores (gemini-2.5-flash, 10 docs)

| Task | Metric | Mean score | Run date |
|---|---|---|---|
| Summarisation | ROUGE-L | 0.244 | 2026-07-03 |
| Summarisation | BERTScore | 0.769 | 2026-07-03 |
| Translation | BLEU | varies by doc (0.08–1.0 on short samples) | 2026-05-17 |

---

## Premium / direct API models

| Key | Model ID | Provider | Input $/1M | Output $/1M | Notes |
|---|---|---|---|---|---|
| `claude-sonnet-4-6` | `claude-sonnet-4-6` | Anthropic | $3.00 | $15.00 | Strong quality; higher cost. Requires `ANTHROPIC_API_KEY`. |
| `gpt-4o-mini` | `gpt-4o-mini` | OpenAI | $0.15 | $0.60 | Good baseline; requires `OPENAI_API_KEY`. |

---

## OpenRouter models (require `OPENROUTER_API_KEY`)

All use `provider_type: openai_compatible` — no extra processor code needed.

| Key | Model ID | Input $/1M | Output $/1M | Smoke-test status | Known quirks |
|---|---|---|---|---|---|
| `llama-3.3-70b` | `meta-llama/llama-3.3-70b-instruct` | $0.10 | $0.32 | Catalog verified | Occasionally verbose on extraction JSON — check `raw_llm_output` if parsing fails. |
| `deepseek-v3` | `deepseek/deepseek-chat` | $0.20 | $0.80 | Catalog verified | **Occasionally returns malformed JSON on the extraction step.** Retry or inspect `outputs/results/`. |
| `qwen3-30b` | `qwen/qwen3-30b-a3b` | $0.12 | $0.50 | Catalog verified | Strong multilingual; good value for translation benchmarks. |
| `qwen2.5-72b` | `qwen/qwen-2.5-72b-instruct` | $0.36 | $0.40 | Catalog verified | Established multilingual baseline. |
| `glm-4-7` | `z-ai/glm-4.7` | $0.40 | $1.75 | Catalog verified | Replaces deprecated `thudm/glm-z1-32b` on OpenRouter. |
| `mistral-small-3.2` | `mistralai/mistral-small-3.2-24b-instruct` | $0.075 | $0.20 | Catalog verified | European model; good for DE→EN translation cost/quality trade-off. |
| `phi-4` | `microsoft/phi-4` | $0.07 | $0.14 | Catalog verified | Small model that punches above its weight on summarisation. |
| `gemma-3-27b` | `google/gemma-3-27b-it` | $0.08 | $0.16 | Catalog verified | Google's open model; competitive on BERTScore at low cost. |

---

## Adding a new model

1. Add an entry under `models.catalog` in `configs/config.yaml`.
2. Set `provider_type` to `gemini`, `claude`, or `openai_compatible`.
3. Set `api_key_env` to the environment variable name in `.env`.
4. Run a 3-doc smoke test from the dashboard with **Benchmark samples** selected.
5. Update this file with scores and any quirks observed.

---

## Prompt versioning

Prompt templates live in `configs/prompts.yaml` with a top-level `version` field.
Every `PipelineResult` and `BenchmarkReport` records the prompt version used at run time.
Edit prompts in the dashboard sidebar or directly in YAML; saved edits snapshot the previous version to `configs/prompt_history/`.

See [docs/VERSIONING.md](VERSIONING.md) for the distinction between package version and prompt version.

---

## LLM-as-Judge (evaluation)

Summarisation evaluation can use **`llm_judge`** — a separate model scores faithfulness, completeness, and coherence (1-5) by reading the source article and generated summary.

| Config key | Default | Notes |
|---|---|---|
| `evaluation.judge_model` | `gpt-4o-mini` | Catalog key; requires the matching `api_key_env` in `.env` |

Judge calls incur API cost and latency — both are recorded per document in the evaluation report metadata. Disable by removing `llm_judge` from `evaluation.metrics.summarisation`.
