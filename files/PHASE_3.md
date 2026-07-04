# Phase 3 — Local Analysis Dashboard

> **Prerequisite:** Phase 2b complete on `main` (v0.2.1 — manifest provenance, CI, RUNBOOK) ✅
> **Primary deliverable:** A Streamlit dashboard that runs locally, lets you pick models and tasks, runs the benchmark, and shows a side-by-side comparison with charts and export.
> **Stack:** Streamlit + existing Python benchmark engine. No deployment, no BYOK, no security plumbing.
> **Time estimate:** ~1 week part-time.

---

## Why not Gradio + Hugging Face Spaces

The original plan was a public BYOK web app. That optimises for *strangers visiting a URL without cloning the repo* — a non-technical audience. The actual audience for this project is engineers and developers who **will** clone the repo, **will** read the README, and **will** run it themselves. For them, a clean local tool is more useful than a deployed demo with session-only key handling and rate limits.

Simpler is better here. A Streamlit dashboard launched with one command is faster to build, more useful day-to-day, and a better README story.

If a public URL becomes desirable after Phase 3 ships — once there's evidence anyone wants it — Streamlit → Gradio is straightforward and becomes Phase 4.

---

## What it delivers

```
clone repo → add API keys to .env → uv run streamlit run app/dashboard.py → open browser
```

The dashboard:

1. **Task picker** — Translation or Summarisation
2. **Model selector** — multi-select from the config catalog
3. **Sample size slider** — 1–30 docs
4. **Run button** — runs the benchmark, shows a per-model progress indicator
5. **Results table** — Model | ROUGE/BLEU | BERTScore | Avg tokens | Total cost | Avg latency
6. **Charts** — bar chart per metric, cost vs. quality scatter
7. **Export** — download JSON or CSV

No API key UI — keys are read from `.env` as they always have been.

---

## Architecture

Minimal. The dashboard is a thin display layer on top of what already exists:

```
app/dashboard.py  (Streamlit)
       │
       ▼
BenchmarkRunner.run(docs, models, task)   ← already built in Phase 2
       │
       ├── PipelineOrchestrator (per model)
       └── Evaluator → EvaluationReport
```

No new abstractions. No new config. The dashboard calls `BenchmarkRunner` exactly the same way the CLI does.

---

## Sub-phases

### Phase 3a — Streamlit MVP

**Goal:** Working dashboard, end-to-end, one command to launch.

| Task | Implementation | Test |
|---|---|---|
| `dashboard-shell` | `app/dashboard.py` with sidebar (task, models, sample size) | Smoke: `import dashboard` |
| `benchmark-runner-call` | Wire sidebar inputs → `BenchmarkRunner.run()` | Mock test: runner called with correct args |
| `results-table` | `st.dataframe` of `BenchmarkReport` | Smoke: table shows model names |
| `metric-bar-chart` | `st.bar_chart` per metric (ROUGE, BLEU, BERTScore) | Visual: chart renders without error |
| `cost-latency-scatter` | `st.scatter_chart` — x: cost per doc, y: BERTScore | Visual smoke |
| `export-buttons` | `st.download_button` for JSON and CSV | Round-trip: downloaded JSON parses back |

**New file:**

```
app/
├── __init__.py
└── dashboard.py
```

**Run command (added to README):**

```bash
uv run streamlit run app/dashboard.py
```

---

### Phase 3b — Static benchmark samples + model catalog

**Goal:** No runtime HuggingFace downloads. Curated dataset committed to repo. More models in catalog.

**Dataset problem today:** EuroParl and CNN/DailyMail loaders download from HF at runtime — slow, flaky, not suitable as a baseline for reproducible comparisons.

**Fix:** Export 30-doc slices once from existing loaders, spot-check them, commit to repo.

```
data/benchmark_samples/
├── translation_de_en.json      # 30 docs: {doc_id, source_language, raw_text, reference}
├── summarisation_en.json       # 30 docs: {doc_id, raw_text, reference_summary}
└── README.md                   # provenance: source, date exported, HF version
```

New loader `src/pipeline/benchmark_sample_loader.py` — reads static JSON, returns `list[DocumentInput]` + ground truths.

**Model catalog expansion (YAML only — no new code):**

```yaml
nemotron-nano:
  provider_type: openai_compatible
  model_id: nvidia/nemotron-nano-9b-v2
  base_url: https://openrouter.ai/api/v1
  api_key_env: OPENROUTER_API_KEY
  pricing: {input_per_1m: 0.04, output_per_1m: 0.04}
```

Target: add 5–10 models via OpenRouter. No new provider code — `OpenAICompatibleProcessor` already handles them.

| Task | Implementation | Test |
|---|---|---|
| `export-benchmark-samples` | One-time script → `data/benchmark_samples/` | Assert JSON schema valid, 30 docs each |
| `benchmark-sample-loader` | `benchmark_sample_loader.py` | Unit: loads samples, returns DocumentInput list |
| `catalog-expansion` | 5–10 OpenRouter model entries in `config.yaml` | Config validation test |
| `dashboard-sample-toggle` | Radio: "Benchmark samples" vs "Live download" | Unit: correct loader used |

---

### Phase 3c — Polish + README

**Goal:** Repo is genuinely useful to someone who clones it for the first time.

| Task | Notes |
|---|---|
| `README overhaul` | Screenshots of dashboard, one-command quickstart, model table, architecture diagram |
| `prompt versioning` | Add `version` field to `configs/prompts.yaml`; thread through `PipelineResult` |
| `session run history` | Last 5 runs stored in `st.session_state`; shown as collapsible in sidebar |
| `docs/MODELS.md` | Which models were tested, smoke-test results, known quirks |

---

## What this does NOT need (was in old Phase 3)

| Removed | Why |
|---|---|
| BYOK session key handling | Keys live in `.env` — no public deployment, no problem |
| Rate limits (max 5 models, max 20 docs) | Your machine, your API budget |
| Key redaction in logs | No public server, no risk |
| HF Spaces deployment | Skip until there's evidence of demand |
| Gradio | Streamlit is simpler; migration later if needed |
| Cost preview before run | Nice-to-have but not critical locally |
| Scanned PDF / OCR | Out of scope entirely |

File upload (`Phase 3b` in old plan) is also dropped. The use case was weak ("compare outputs without ground truth") and adds complexity with no payoff.

---

## Git workflow

Same as always — branch off `dev`, one branch per sub-phase:

| Branch | Scope |
|---|---|
| `feature/phase3a-dashboard` | Streamlit shell + BenchmarkRunner wiring + charts + export |
| `feature/phase3b-samples` | Static benchmark samples + benchmark-sample-loader + catalog expansion |
| `feature/phase3c-polish` | README screenshots, prompt versioning, run history, MODELS.md |

Final Phase 3 completion: PR **`dev` → `main`**, tag `v0.3.0`.

---

## Version map

| Version | Item |
|---|---|
| `0.2.1` | Phase 2b complete — manifest, truncation, CI, RUNBOOK ✅ |
| `0.2.2` | Phase 3a — Streamlit dashboard MVP |
| `0.2.3` | Phase 3b — static benchmark samples + catalog expansion |
| `0.3.0` | Phase 3c — polish + README; promote to `main` |

---

## Phase 3 → Phase 4 gate

**Do not start Phase 4 until:**

- [ ] Dashboard runs end-to-end with at least 2 models
- [ ] Static benchmark samples committed and loader tested
- [ ] README shows a screenshot and a one-command quickstart
- [ ] CI green on `main` after Phase 3c merge

Then say **"start phase 4"** — which is where a public URL, HF Spaces, or any deployment story lives if it's warranted.

---

## Recommended next action

Say **"start phase 3a"** to begin `feature/phase3a-dashboard`.
