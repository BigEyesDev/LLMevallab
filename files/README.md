# 🧠 Multilingual Document Intelligence — LLM Extraction, Translation & Evaluation Framework

> **A hands-on, portfolio-grade project for learning and demonstrating LLM evaluation in production systems.**
> Built incrementally across three phases — from working prototype to published, world-ready showcase.

---

## 📌 Project Summary

This project builds an end-to-end pipeline that:

1. **Ingests** multilingual documents (German, French, Spanish, and more)
2. **Extracts** structured information — entities, dates, deadlines, key clauses
3. **Translates** non-English content to English using LLMs
4. **Summarises** documents into concise, structured outputs
5. **Evaluates** all of the above with industry-standard metrics and a live dashboard

The system is **model-agnostic by design** — you can swap or add LLM providers (Gemini, Claude, OpenAI, open-source models) by adding a single file. No pipeline rewriting required.

---

## 🗂️ Repository Structure

```
multilingual-doc-intelligence/
│
├── README.md                          ← You are here (master overview)
│
├── data/
│   ├── raw/                           ← Raw downloaded documents (EuroParl, etc.)
│   ├── processed/                     ← Cleaned, structured documents
│   └── ground_truth/                  ← Human-annotated references for evaluation
│
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── base_processor.py          ← Abstract base class (model-agnostic interface)
│   │   └── models.py                  ← Pydantic data models (DocumentInput, ExtractionResult, etc.)
│   │
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── gemini_processor.py        ← Gemini implementation
│   │   ├── claude_processor.py        ← Claude implementation (Phase 2)
│   │   ├── openai_processor.py        ← OpenAI implementation (Phase 2)
│   │   └── opensource_processor.py   ← DeepSeek / Qwen etc. (Phase 2)
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── orchestrator.py            ← Runs the full pipeline (extract → translate → summarise)
│   │   └── data_loader.py             ← Dataset downloading and preprocessing
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                 ← BLEU, ROUGE, BERTScore, COMET, LLM-as-Judge
│   │   ├── evaluator.py               ← Runs evaluation across all metrics
│   │   └── benchmark.py              ← Multi-model comparison runner
│   │
│   └── dashboard/
│       ├── __init__.py
│       └── app.py                     ← Streamlit dashboard (Phase 3)
│
├── notebooks/
│   ├── 01_extraction_translation_summarisation.ipynb   ← Phase 1 walkthrough
│   ├── 02_evaluation_framework.ipynb                   ← Phase 2 deep-dive
│   └── 03_multi_model_benchmark.ipynb                  ← Phase 3 comparison
│
├── configs/
│   ├── config.yaml                    ← Global config (models, paths, params)
│   └── prompts.yaml                   ← All LLM prompts (centralised, versioned)
│
├── outputs/
│   ├── results/                       ← JSON/CSV outputs per run
│   └── reports/                       ← Auto-generated evaluation reports
│
├── tests/
│   ├── test_processors.py
│   ├── test_metrics.py
│   └── test_pipeline.py
│
├── requirements.txt
├── .env.example                       ← API key template (never commit .env)
└── .gitignore
```

---

## 🗃️ Dataset: EuroParl Corpus

We use the **[EuroParl Parallel Corpus](https://www.statmt.org/europarl/)** as our primary dataset.

| Property | Detail |
|---|---|
| **Source** | European Parliament proceedings |
| **Languages** | 21 EU languages including German (de), French (fr), Spanish (es), Italian (it) |
| **Format** | Plain text, sentence-aligned |
| **Why it's great** | Real-world formal documents, contains dates/deadlines/legal language, openly licensed, well-studied |
| **Size** | ~50MB per language pair (we use a small subset) |
| **License** | Public domain |

**Supplementary dataset:** For more realistic document extraction scenarios, we also use a small curated set of German-language news articles from the **[CC-100 corpus](https://huggingface.co/datasets/cc100)** available via HuggingFace.

---

## 🔬 Evaluation Metrics (Industry Standard — 2026)

| # | Metric | What It Measures | Primary Use |
|---|---|---|---|
| 1 | **BLEU** | N-gram overlap between generated and reference translation | Translation accuracy |
| 2 | **ROUGE-L** | Longest common subsequence recall | Summarisation quality |
| 3 | **BERTScore** | Semantic similarity using contextual embeddings | Translation + extraction |
| 4 | **COMET** | Neural MT quality estimation (reference-based) | Translation (state-of-art) |
| 5 | **LLM-as-Judge** | GPT/Claude rates output on faithfulness, completeness, coherence | Extraction + summarisation |

---

## 🤖 Supported Models

The architecture is designed so each model is a **drop-in provider**. Adding a new model requires only:

1. Creating one new file in `src/providers/`
2. Inheriting from `BaseDocumentProcessor`
3. Implementing three methods: `extract()`, `translate()`, `summarise()`

| Model | Provider File | Status |
|---|---|---|
| Gemini 1.5 Pro | `gemini_processor.py` | ✅ Phase 1 |
| Claude Sonnet | `claude_processor.py` | 🔜 Phase 2 |
| GPT-4o | `openai_processor.py` | 🔜 Phase 2 |
| DeepSeek / Qwen | `opensource_processor.py` | 🔜 Phase 2 |

---

## 🗺️ Three-Phase Roadmap

### Phase 1 — Working Prototype (Current)
**Goal:** Build the full pipeline with one model (Gemini), demonstrate extraction, translation, and summarisation on real documents, and set up evaluation scaffolding.

**Deliverables:**
- `src/core/` — base architecture and data models
- `src/providers/gemini_processor.py` — first working model
- `src/pipeline/` — data loading and orchestration
- `src/evaluation/metrics.py` — BLEU, ROUGE, BERTScore implemented
- `notebooks/01_extraction_translation_summarisation.ipynb`
- Dataset downloaded and processed

**What you learn in Phase 1:**
- How to design model-agnostic LLM pipelines with abstract base classes
- How to write structured LLM prompts that return parseable JSON
- What BLEU, ROUGE, and BERTScore actually measure (with intuitive examples)
- How to handle multilingual text in Python

---

### Phase 2 — Multi-Model Evaluation
**Goal:** Plug in Claude, OpenAI, and an open-source model. Run head-to-head benchmarks. Add COMET and LLM-as-Judge. Build evaluation reports.

**Deliverables:**
- `src/providers/` — all model implementations
- `src/evaluation/benchmark.py` — side-by-side comparisons
- `src/evaluation/evaluator.py` — full evaluation runner
- `notebooks/02_evaluation_framework.ipynb`
- `notebooks/03_multi_model_benchmark.ipynb`
- Auto-generated reports in `outputs/reports/`

**What you learn in Phase 2:**
- COMET — the current industry standard for MT evaluation
- LLM-as-Judge — how to use one LLM to evaluate another
- How different models perform on the same extraction/translation task
- Statistical significance in benchmark comparisons

---

### Phase 3 — Dashboard & Public Showcase
**Goal:** Build a Streamlit dashboard where anyone can upload a document in any language and see extraction, translation, summarisation, and evaluation scores live. Publish to Substack/LinkedIn.

**Deliverables:**
- `src/dashboard/app.py` — full interactive dashboard
- Live model selector (choose which model to run)
- Side-by-side output comparison
- Real-time metric scores and visualisations
- Written article summarising findings

**What you learn in Phase 3:**
- Building production-facing LLM interfaces
- How to communicate evaluation results to non-technical audiences
- Deployment patterns for LLM apps

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/multilingual-doc-intelligence.git
cd multilingual-doc-intelligence

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up API keys
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 5. Download dataset
python src/pipeline/data_loader.py

# 6. Run Phase 1 pipeline
python src/pipeline/orchestrator.py --model gemini --input data/processed/sample.json
```

---

## 📐 Design Principles

This project is built to **senior engineering standards**:

| Principle | How It's Applied |
|---|---|
| **Single Responsibility** | Each module does one thing (extraction, evaluation, loading) |
| **Open/Closed** | Add new models without modifying existing code |
| **Dependency Inversion** | Pipeline depends on abstract `BaseDocumentProcessor`, not concrete models |
| **Fail Loudly** | Custom exceptions, typed returns, no silent failures |
| **Configuration over Code** | All prompts and parameters live in `configs/`, not hardcoded |
| **Reproducibility** | Runs are logged with model version, config snapshot, and timestamp |

---

## 📎 References & Further Reading

- [EuroParl Corpus](https://www.statmt.org/europarl/) — Philipp Koehn, 2005
- [BLEU Score Paper](https://aclanthology.org/P02-1040/) — Papineni et al., 2002
- [BERTScore Paper](https://arxiv.org/abs/1904.09675) — Zhang et al., 2019
- [COMET](https://github.com/Unbabel/COMET) — Rei et al., 2020
- [DeepEval Framework](https://github.com/confident-ai/deepeval) — LLM-as-Judge and more
- [LLM-as-Judge](https://arxiv.org/abs/2306.05685) — Zheng et al., 2023

---

*Built with curiosity, experiments, and a lot of trial and error. — Phase 1 started May 2026.*
