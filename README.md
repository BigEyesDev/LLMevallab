# 🧠 Multilingual Document Intelligence — LLM Extraction, Translation & Evaluation Framework

> **A hands-on project for learning and demonstrating LLM evaluation in production systems.**
> Building incrementally across three phases — from working prototype to published, world-ready showcase.
> Code is written 60-70% by Claude, rest by me - real user. Agenda is to learn more about eval metrics for LLMs and bring more clarity over the course of time!
>
> **Author:** [Akash Sachdeva (@BigEyesDev)](https://github.com/BigEyesDev) — Senior Engineer focused on ML, MLOps, and applied LLMs.

**Version:** `0.2.1` — see [CHANGELOG.md](CHANGELOG.md) and [docs/VERSIONING.md](docs/VERSIONING.md).

---

## Pipeline Overview

![Pipeline Overview](docs/assets/pipeline_overview.png)

## 📌 Project Summary

This project builds an end-to-end pipeline that:

1. **Ingests** multilingual documents (German, French, Spanish, and more)
2. **Extracts** structured information — entities, dates, deadlines, key clauses
3. **Translates** non-English content to English using LLMs
4. **Summarises** documents into concise, structured outputs
5. **Evaluates** all of the above with industry-standard metrics and a live dashboard

The system is **model-agnostic by design** i.e. you can swap or add LLM providers (Gemini, Claude, OpenAI, open-source models) by adding a single file. No pipeline rewriting required.

---

## Dashboard (Phase 3a)

```bash
uv run streamlit run app/dashboard.py
```

Opens a local Streamlit UI where you can pick a task (translation or summarisation), select models from the catalog, set a sample size, run the benchmark, and explore quality metrics, cost, and latency side-by-side. Export results as JSON or CSV.

**Prerequisites:** API keys in `.env` (see `.env.example`). Datasets prepared via `python main.py`.

---


