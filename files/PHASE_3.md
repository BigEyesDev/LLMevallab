# Phase 3 — Local Analysis Dashboard

> **Prerequisite:** Phase 2b complete on `main` (v0.2.1 — manifest provenance, CI, RUNBOOK) ✅
> **Primary deliverable:** A Streamlit dashboard that runs locally, lets you pick models and tasks, runs the benchmark, and shows a side-by-side comparison with charts and export.
> **Stack:** Streamlit + existing Python benchmark engine. No deployment, no BYOK, no security plumbing.

---

## Why not Gradio + Hugging Face Spaces

The original plan was a public BYOK web app. That optimises for *strangers visiting a URL without cloning the repo* — a non-technical audience. The actual audience for this project is engineers and developers who **will** clone the repo, **will** read the README, and **will** run it themselves. For them, a clean local tool is more useful than a deployed demo with session-only key handling and rate limits.

Simpler is better here. A Streamlit dashboard launched with one command is faster to build, more useful day-to-day, and a better README story.

Public deployment (Docker, HF Spaces) is deferred to Phase 4 — only if there is evidence of real demand.

---

## HuggingFace — two separate roles

| Role | What | Status |
|---|---|---|
| **HF Datasets (data source)** | Python library to download EuroParl + CNN/DailyMail benchmark data | Used in `python main.py`. One-time download, cached locally by `datasets` lib. Replaced by static samples for offline use. |
| **HF Spaces (deployment platform)** | Host dashboard as a public web app | **Removed from Phase 3.** Deferred to Phase 4. |

---

## Data layer architecture

Two tiers — the dashboard exposes both via a sidebar radio:

```
Tier 1 — Benchmark samples (built-in, always available, no setup required)
  data/benchmark_samples/
  ├── translation_de_en.json     30 docs committed to git
  ├── summarisation_en.json      30 docs committed to git
  └── README.md                  provenance: source, HF version, export date

Tier 2 — Full HF dataset (one-time download, up to 100+ docs)
  data/processed/
  ├── europarl/europarl_de-en_Ndocs.json     after: python main.py
  └── cnn_dailymail/cnn_dailymail_Ndocs.json after: python main.py
```

HF `datasets` library caches to `~/.cache/huggingface/` — subsequent runs are instant. A user who wants more than 30 docs runs `python main.py` once and never again.

---

## Architecture

```
app/dashboard.py  (Streamlit)
       │
       ├── BenchmarkSampleLoader   ← Tier 1: reads static JSON (offline)
       └── EuroParlLoader /        ← Tier 2: reads HF-downloaded processed JSON
           CNNDailyMailLoader
       │
       ▼
BenchmarkRunner.run(docs, models, task, documents=...)
       │
       ├── PipelineOrchestrator (per model)
       └── Evaluator → EvaluationReport
```

---

## Sub-phases

### Phase 3a — Streamlit MVP ✅ COMPLETE (v0.2.2)

**Delivered:**

| Feature | File | Notes |
|---|---|---|
| Dashboard shell | `app/dashboard.py` | Task picker, model multi-select, doc selector grid |
| BenchmarkRunner wiring | `app/dashboard.py` | Per-model progress bar + step log |
| Results table | `app/dashboard.py` | `st.dataframe`, all metrics + cost + latency |
| Quality metric bar charts | `app/dashboard.py` | Altair, per-model colours, score labels |
| Cost vs. quality scatter | `app/dashboard.py` | Altair, labelled points, size = latency |
| Export buttons | `app/dashboard.py` | JSON + CSV download |
| Run cache | `app/dashboard.py` | Session-state cache keyed on (task, models, doc IDs) |
| Doc card grid | `app/dashboard.py` | 3-column cards, checkboxes, All/None/Random N |
| Pre-run context | `app/dashboard.py` | Task info, metric explainers, model pricing table, dataset preview |
| Key takeaways | `app/dashboard.py` | Auto-generated plain-English interpretation of results |
| `documents` param | `src/evaluations/benchmark.py` | Optional — lets dashboard inject specific doc list |

---

### Phase 3b — Static benchmark samples + model catalog ✅ COMPLETE (v0.3.0)

**Goal:** Zero-setup offline benchmark data. More models. Dashboard data source toggle.

#### `export-benchmark-samples`

One-time export script (`scripts/export_benchmark_samples.py`) reads the existing processed
JSON files and writes curated 30-doc slices to `data/benchmark_samples/`.

```
data/benchmark_samples/
├── translation_de_en.json      # 30 docs: {doc_id, source_language, raw_text, metadata.reference_translation}
├── summarisation_en.json       # 30 docs: {doc_id, raw_text, metadata.reference_summary}
└── README.md                   # provenance: source dataset, HF version, export date, doc_ids
```

#### `benchmark-sample-loader`

New file: `src/pipeline/benchmark_sample_loader.py`

```python
class BenchmarkSampleLoader:
    """Reads committed static JSON benchmark samples. No network required."""
    def load(self, task: str) -> list[DocumentInput]: ...
    def ground_truth(self, task: str) -> dict[str, str]: ...
```

Follows the same interface as `EuroParlDataLoader` and `CNNDailyMailLoader`.
Returns `list[DocumentInput]` + a ground-truth dict — the evaluator receives both unchanged.

#### `catalog-expansion`

Pure YAML — no new code. Add to `configs/config.yaml`:

| Key | Model ID | Provider | Notes |
|---|---|---|---|
| `qwen3-30b` | `qwen/qwen3-30b-a3b` | OpenRouter | Strong multilingual, cheap |
| `qwen2.5-72b` | `qwen/qwen-2.5-72b-instruct` | OpenRouter | Established multilingual baseline |
| `glm-4-7` | `z-ai/glm-4.7` | OpenRouter | Zhipu GLM-4 flagship (replaces deprecated glm-z1-32b) |
| `mistral-small-3.2` | `mistralai/mistral-small-3.2-24b-instruct` | OpenRouter | European model, strong multilingual |
| `phi-4` | `microsoft/phi-4` | OpenRouter | Small but punches above its weight |
| `gemma-3-27b` | `google/gemma-3-27b-it` | OpenRouter | Google open model |

All use `provider_type: openai_compatible` — `OpenAICompatibleProcessor` handles them with zero new code.
Claude stays in catalog for completeness but is not the default. Default remains `gemini-2.5-flash`.

#### `dashboard-sample-toggle`

Sidebar radio added to `app/dashboard.py`:

```
Data source
  ● Benchmark samples   (30 docs, offline, always available)
  ○ Full dataset        (requires python main.py, up to 100+ docs)
```

When "Full dataset" is selected and `data/processed/` does not exist: show a clear inline
error — `"Run python main.py first to download the full dataset."` No silent failure.

**Task table:**

| Task | Implementation | Test |
|---|---|---|
| `export-benchmark-samples` | `scripts/export_benchmark_samples.py` | Assert JSON schema valid, 30 docs each |
| `benchmark-sample-loader` | `src/pipeline/benchmark_sample_loader.py` | Unit: loads samples, returns `DocumentInput` list |
| `catalog-expansion` | `configs/config.yaml` | Config validation test |
| `dashboard-sample-toggle` | `app/dashboard.py` | Unit: correct loader called per radio selection |

**New / changed files:**

```
scripts/
└── export_benchmark_samples.py   ← one-time export (run once, commit output)

data/benchmark_samples/
├── translation_de_en.json        ← committed
├── summarisation_en.json         ← committed
└── README.md                     ← committed

src/pipeline/
└── benchmark_sample_loader.py    ← new loader

configs/config.yaml               ← 6 new model entries

app/dashboard.py                  ← data source radio toggle
```

---

### Phase 3c — Prompt editor + polish

**Goal:** Repo is genuinely useful to someone who clones it for the first time. Prompts are visible, editable, and versioned without leaving the dashboard.

#### Prompt viewer + editor

New sidebar section in `app/dashboard.py`:

1. **View** — expander shows the current system prompt and user template for the selected task (read from `configs/prompts.yaml`)
2. **Edit** — `st.text_area` lets the user modify either field inline
3. **Save as new version** — button writes the edited prompts back to `configs/prompts.yaml` with an incremented `version` field and snapshots the old version to `configs/prompt_history/`

```
configs/
├── prompts.yaml                                     ← active prompts (always current)
└── prompt_history/
    ├── v1_20260704_090000_original.yaml
    ├── v2_20260706_143000_shorter_summary.yaml
    └── ...
```

Each snapshot file contains: version number, timestamp, author note (user types a short description), and full prompt content.

Every `PipelineResult` records the prompt version used — results are always traceable to the exact prompt that produced them.

#### Session run history

Last 5 benchmark runs stored in `st.session_state`. Shown as a collapsible section in the sidebar: timestamp, task, models used, top metric scores. Click any past run to restore its report to the main view.

#### README overhaul

- Screenshot of dashboard (pre-run context + post-run results)
- One-command quickstart (`uv run streamlit run app/dashboard.py`)
- Model catalog table with pricing
- Architecture diagram

#### `docs/MODELS.md`

Which models were tested, smoke-test scores, known quirks (e.g., "DeepSeek-V3 occasionally returns malformed JSON on extraction step").

#### `Makefile`

```makefile
setup:     # uv sync + python main.py (full HF dataset download)
run:       # uv run streamlit run app/dashboard.py
test:      # uv run pytest
```

**Task table:**

| Task | Notes |
|---|---|
| `prompt-viewer` | Read `prompts.yaml`, render in sidebar expander |
| `prompt-editor` | `st.text_area` edit + save, write to `prompts.yaml` |
| `prompt-versioning` | Snapshot to `configs/prompt_history/`, increment version field |
| `prompt-version-in-result` | Thread version string through `PipelineResult` |
| `session-run-history` | Last 5 runs in session_state, collapsible sidebar |
| `README-overhaul` | Screenshot, quickstart, model table, diagram |
| `docs/MODELS.md` | Tested models, known quirks |
| `Makefile` | `setup`, `run`, `test` targets |

---

## What this does NOT need

| Removed | Why |
|---|---|
| BYOK session key handling | Keys live in `.env` — no public deployment, no problem |
| Rate limits (max 5 models, max 20 docs) | Your machine, your API budget |
| Key redaction in logs | No public server, no risk |
| HF Spaces deployment | Phase 4 only — if there is evidence of demand |
| Gradio | Streamlit is simpler; migration later if needed |
| Cost preview before run | Nice-to-have but not critical locally |
| Scanned PDF / OCR | Out of scope entirely |
| Docker | Phase 4 — once project is stable enough for external users |

---

## Git workflow

| Branch | Scope |
|---|---|
| `feature/phase3a-dashboard` | ✅ merged |
| `feature/phase3b-samples` | ✅ merged |
| `feature/phase3c-polish` | Prompt editor/versioning, run history, README, Makefile |

Final Phase 3 completion: PR **`dev` → `main`**, tag `v0.3.0`.

---

## Version map

| Version | Item |
|---|---|
| `0.2.1` | Phase 2b complete — manifest, truncation, CI, RUNBOOK ✅ |
| `0.2.2` | Phase 3a — Streamlit dashboard MVP ✅ |
| `0.2.3` | (skipped — folded into 0.3.0) |
| `0.3.0` | Phase 3a + 3b — dashboard, offline samples, catalog expansion ✅ |

---

## Phase 3 → Phase 4 gate

**Do not start Phase 4 until:**

- [x] Dashboard runs end-to-end with at least 2 models
- [x] Static benchmark samples committed and loader tested
- [ ] README shows a screenshot and a one-command quickstart
- [ ] CI green on `main` after Phase 3c merge

Phase 4 is where Docker, a public URL, HF Spaces, or any deployment story lives — if warranted.

---

## Recommended next action

Say **"implement phase 3c"** to begin prompt editor, run history, and README polish.
