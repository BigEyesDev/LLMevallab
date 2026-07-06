"""Streamlit dashboard — thin display layer over BenchmarkRunner."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.core.config import get_model_catalog, get_processed_path, load_config
from src.core.models import BenchmarkReport, DocumentInput, ModelBenchmarkResult
from src.evaluations.benchmark import BenchmarkRunner
from src.pipeline.benchmark_sample_loader import BenchmarkSampleLoader
from src.pipeline.orchestrator import PipelineTask, load_prompts
from src.pipeline.prompt_manager import (
    get_prompt_version,
    get_task_prompts,
    save_prompt_version,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_METRIC_DISPLAY: dict[str, str] = {
    "bleu": "BLEU",
    "rouge": "ROUGE-L",
    "rouge_l": "ROUGE-L",   # evaluator emits "rouge_l"; keep "rouge" for back-compat
    "bertscore": "BERTScore",
}

_METRIC_DESCRIPTIONS: dict[str, dict] = {
    "BLEU": {
        "short": "N-gram word overlap vs. reference. Higher = closer word-for-word match.",
        "range": "0 – 1",
        "good": "> 0.30 for translation",
        "detail": (
            "Counts how many short word sequences (n-grams) appear in both the model output "
            "and the human reference. Fast to compute and widely used, but does not capture "
            "paraphrases — a perfect synonym scores zero."
        ),
    },
    "ROUGE-L": {
        "short": "Longest common subsequence vs. reference. Captures sentence structure.",
        "range": "0 – 1",
        "good": "> 0.25 for news summarisation",
        "detail": (
            "Finds the longest sequence of words that appear in both texts (in order, not "
            "necessarily contiguous). Rewards outputs that preserve the flow and structure of "
            "the reference. The standard metric for news summarisation benchmarks."
        ),
    },
    "BERTScore": {
        "short": "Semantic similarity via BERT embeddings. Robust to paraphrasing.",
        "range": "0 – 1",
        "good": "> 0.70 (F1, DeBERTa model)",
        "detail": (
            "Embeds each token with a pre-trained BERT model and computes cosine similarity "
            "between output and reference. A paraphrase that conveys the same meaning "
            "scores highly even if no words overlap. Most correlated with human judgment."
        ),
    },
}

_TASK_META: dict[str, dict] = {
    "translation": {
        "label": "Translation (DE → EN)",
        "description": (
            "Each model receives a German document and produces an English translation. "
            "The output is compared to a human-authored reference translation."
        ),
        "dataset": "EuroParl (EU Parliament proceedings)",
        "dataset_key": "europarl",
        "metrics": ["BLEU", "BERTScore"],
        "preview_field": "raw_text",
        "preview_label": "Source (German)",
        "preview_ref_field": "reference_translation",
        "preview_ref_label": "Reference (English)",
    },
    "summarisation": {
        "label": "Summarisation (EN)",
        "description": (
            "Each model receives a full news article and produces a concise summary. "
            "The output is scored against a journalist-written reference summary."
        ),
        "dataset": "CNN / DailyMail (news articles)",
        "dataset_key": "cnn_dailymail",
        "metrics": ["ROUGE-L", "BERTScore"],
        "preview_field": "raw_text",
        "preview_label": "Article",
        "preview_ref_field": "reference_summary",
        "preview_ref_label": "Reference summary",
    },
}

_PALETTE = ["#4F8EF7", "#F7884F", "#4FF7A0", "#F74F92", "#B44FF7", "#F7D24F"]
_RUNNABLE_TASKS = [PipelineTask.SUMMARISATION.value, PipelineTask.TRANSLATION.value]
_MAX_RUN_HISTORY = 5

# Rough characters-per-token estimates used for pre-run cost hints.
# GPT-style tokenisers average ~4 chars/token for English.
# German is morphologically richer — inflected forms and compounds mean more
# characters per token, typically ~3 chars/token.
_CHARS_PER_TOKEN: dict[str, float] = {"de": 3.0, "en": 4.0}
_DEFAULT_CHARS_PER_TOKEN = 4.0


# ---------------------------------------------------------------------------
# Resource & data loading
# ---------------------------------------------------------------------------

@st.cache_data
def _load_resources(_config_mtime: float, _prompts_mtime: float) -> tuple[dict, dict]:
    """Load config and prompts, keyed on file mtimes so edits are picked up automatically."""
    config = load_config("configs/config.yaml")
    prompts = load_prompts()
    return config, prompts


def _resource_mtimes() -> tuple[float, float]:
    """Return modification times of both config files (used as cache-busting keys)."""
    config_path = Path("configs/config.yaml")
    prompts_path = Path("configs/prompts.yaml")
    return (
        config_path.stat().st_mtime if config_path.exists() else 0.0,
        prompts_path.stat().st_mtime if prompts_path.exists() else 0.0,
    )


@st.cache_data
def _load_all_docs(_config: dict, task: str, max_docs: int = 30) -> list[dict]:
    """Load up to max_docs from the processed dataset for this task (cached per task)."""
    meta = _TASK_META.get(task, {})
    try:
        path = Path(get_processed_path(_config, meta.get("dataset_key", "")))
        if path.exists():
            return json.loads(path.read_text())[:max_docs]
    except Exception:
        pass
    return []


@st.cache_data
def _load_benchmark_samples(task: str) -> list[dict]:
    """Load static benchmark samples committed to git (no network required)."""
    try:
        loader = BenchmarkSampleLoader()
        return [doc.model_dump() for doc in loader.load(task)]
    except (FileNotFoundError, ValueError):
        return []


def _model_colors(model_keys: list[str]) -> dict[str, str]:
    """Assign a consistent hex colour to each model key."""
    return {k: _PALETTE[i % len(_PALETTE)] for i, k in enumerate(model_keys)}


def _token_cost_hint(
    raw_text: str,
    source_language: str,
    truncation_limit: int,
    model_keys: list[str],
    catalog: dict,
) -> str:
    """
    Return a one-line pre-run estimate for a single document:
      ~N tok [✂ if truncated] · ~$X [or $X–$Y range across selected models]

    Token count is estimated from character length using language-specific
    chars-per-token ratios. Cost is based on input_per_1m from config.yaml —
    these are estimates only (see pricing disclaimer in the model table above).
    """
    chars_orig = len(raw_text)
    chars_sent = min(chars_orig, truncation_limit)
    cpt = _CHARS_PER_TOKEN.get(source_language, _DEFAULT_CHARS_PER_TOKEN)
    est_tokens = max(1, round(chars_sent / cpt))

    tok_label = f"~{est_tokens:,} tok"
    if chars_sent < chars_orig:
        tok_label += f" ✂ ({chars_orig:,} chars)"

    if not model_keys:
        return tok_label

    costs = [
        est_tokens * catalog[k].get("pricing", {}).get("input_per_1m", 0) / 1_000_000
        for k in model_keys
        if k in catalog and catalog[k].get("pricing", {}).get("input_per_1m", 0) > 0
    ]
    if not costs:
        return tok_label

    lo, hi = min(costs), max(costs)
    cost_label = f"~${lo:.5f}" if abs(hi - lo) < 1e-9 else f"~${lo:.5f}–${hi:.5f}"
    return f"{tok_label} · {cost_label}"


# ---------------------------------------------------------------------------
# Doc selection — session-state helpers
# ---------------------------------------------------------------------------

def _sync_doc_selection(all_docs: list[dict], sample_size: int, task: str) -> None:
    """Reset checkbox states to the first sample_size docs when task or size changes."""
    prev_task = st.session_state.get("_doc_task")
    prev_size = st.session_state.get("_doc_sample_size")
    if prev_task != task or prev_size != sample_size:
        st.session_state["_doc_task"] = task
        st.session_state["_doc_sample_size"] = sample_size
        st.session_state.pop("_show_all_docs", None)
        for i, doc in enumerate(all_docs):
            st.session_state[f"_doc_cb_{doc['doc_id']}"] = (i < sample_size)


def _get_selected_docs(all_docs: list[dict]) -> list[dict]:
    return [d for d in all_docs if st.session_state.get(f"_doc_cb_{d['doc_id']}", False)]


def _cache_key(task: str, model_keys: list[str], doc_ids: list[str]) -> tuple:
    return (task, tuple(sorted(model_keys)), tuple(sorted(doc_ids)))


def _clear_run_state() -> None:
    """Drop stored results so the pre-run setup view is shown again."""
    for key in ("report", "report_task", "_from_cache"):
        st.session_state.pop(key, None)


def _selected_doc_ids(all_docs: list[dict]) -> list[str]:
    return [d["doc_id"] for d in _get_selected_docs(all_docs)]


def _top_metric_scores(report: BenchmarkReport) -> dict[str, float]:
    """Best score per quality metric across all models in a report."""
    scores: dict[str, float] = {}
    for result in report.results:
        for metric, value in result.quality_metrics.items():
            label = _METRIC_DISPLAY.get(metric, metric.upper())
            scores[label] = max(scores.get(label, float("-inf")), value)
    return scores


def _make_history_entry(
    report: BenchmarkReport,
    model_keys: list[str],
    prompt_version: str,
) -> dict:
    return {
        "timestamp": report.run_timestamp,
        "task": report.task,
        "models": list(model_keys),
        "doc_count": report.sample_size,
        "prompt_version": prompt_version,
        "top_scores": _top_metric_scores(report),
        "report": report,
    }


def _append_run_history(entry: dict) -> None:
    history = st.session_state.setdefault("_run_history", [])
    history.insert(0, entry)
    st.session_state["_run_history"] = history[:_MAX_RUN_HISTORY]


def _history_button_label(entry: dict) -> str:
    ts = entry["timestamp"][:16].replace("T", " ")
    models = ", ".join(entry["models"][:2])
    if len(entry["models"]) > 2:
        models += f" +{len(entry['models']) - 2}"
    return f"{ts} · {entry['task']} · {models}"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

_DATA_SOURCE_SAMPLES = "Benchmark samples"
_DATA_SOURCE_FULL = "Full dataset"


def _inject_sidebar_styles() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] .st-key-sidebar_configure_new_run button[kind="secondary"],
        [data-testid="stSidebar"] .st-key-sidebar_configure_new_run button[kind="primary"] {
            background-color: #22c55e !important;
            color: #ffffff !important;
            border: 1px solid #16a34a !important;
        }
        [data-testid="stSidebar"] .st-key-sidebar_configure_new_run button:hover:not(:disabled) {
            background-color: #16a34a !important;
            border-color: #15803d !important;
            color: #ffffff !important;
        }
        [data-testid="stSidebar"] .st-key-sidebar_configure_new_run button:disabled {
            background-color: #1f2937 !important;
            color: #6b7280 !important;
            border-color: #374151 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _has_stored_report() -> bool:
    return st.session_state.get("report") is not None


def _sidebar_settings(config: dict) -> tuple[str, list[str], int]:
    catalog: dict = get_model_catalog(config)
    default_model = config.get("models", {}).get("default", list(catalog.keys())[0])

    with st.sidebar:
        _inject_sidebar_styles()
        st.header("Benchmark Settings")

        task: str = st.selectbox("Task", _RUNNABLE_TASKS)

        model_keys: list[str] = st.multiselect(
            "Models",
            options=list(catalog.keys()),
            default=[default_model] if default_model in catalog else [list(catalog.keys())[0]],
        )

        sample_size: int = st.slider(
            "Default selection (docs)",
            min_value=1,
            max_value=30,
            value=5,
            help="Sets the initial document selection. You can override it in the dataset grid below.",
        )

    return task, model_keys, sample_size


def _sidebar_data_source() -> str:
    with st.sidebar:
        st.divider()
        data_source: str = st.radio(
            "Data source",
            options=[_DATA_SOURCE_SAMPLES, _DATA_SOURCE_FULL],
            index=0,
            captions=[
                "30 docs committed to git — works offline, zero setup.",
                "100+ docs from HuggingFace — run `python main.py` first.",
            ],
            help=(
                "**Benchmark samples** — static JSON files committed to the repo. "
                "Always available, no internet needed, ideal for quick comparisons.\n\n"
                "**Full dataset** — downloads EuroParl / CNN-DailyMail via the HuggingFace "
                "`datasets` library. Gives you more documents for a deeper eval. "
                "Requires running `python main.py` once to cache locally."
            ),
        )
        st.caption("API keys from `.env`.")
    return data_source


def _sidebar_prompt_editor(task: str, prompts: dict) -> None:
    """View and edit task prompts; save creates a versioned snapshot."""
    version = get_prompt_version(prompts)
    task_prompts = get_task_prompts(prompts, task)

    with st.sidebar:
        st.divider()
        with st.expander(f"Prompts (v{version})", expanded=False):
            st.caption(f"**{task}** templates from `configs/prompts.yaml`")
            st.markdown("**System**")
            st.code(task_prompts["system"], language=None)
            st.markdown("**User template**")
            st.code(task_prompts["user"], language=None)

            st.markdown("**Edit**")
            system = st.text_area(
                "System prompt",
                value=task_prompts["system"],
                height=120,
                key=f"_prompt_sys_{task}",
            )
            user = st.text_area(
                "User template",
                value=task_prompts["user"],
                height=180,
                key=f"_prompt_user_{task}",
            )
            note = st.text_input(
                "Version note",
                placeholder="e.g. shorter summary",
                key="_prompt_note",
            )
            if st.button("Save as new version", key="_prompt_save", use_container_width=True):
                new_version = save_prompt_version(task, system, user, note)
                _load_resources.clear()
                st.toast(f"Saved prompt v{new_version}", icon="💾")
                st.rerun()


def _sidebar_run_history() -> None:
    """Collapsible list of the last five benchmark runs in this session."""
    history: list[dict] = st.session_state.get("_run_history", [])
    if not history:
        return

    with st.sidebar:
        st.divider()
        with st.expander("Recent runs", expanded=False):
            st.caption("Click a run to restore its results in the main view.")
            for i, entry in enumerate(history):
                if st.button(
                    _history_button_label(entry),
                    key=f"_hist_{i}",
                    use_container_width=True,
                ):
                    st.session_state["report"] = entry["report"]
                    st.session_state["report_task"] = entry["task"]
                    st.session_state["_from_cache"] = False
                    st.rerun()
                score_parts = [
                    f"{metric} {value:.3f}" for metric, value in entry["top_scores"].items()
                ]
                st.caption(
                    f"v{entry['prompt_version']} · {entry['doc_count']} docs"
                    + (f" · {' · '.join(score_parts)}" if score_parts else "")
                )


def _sidebar_run_button(model_keys: list[str]) -> bool:
    with st.sidebar:
        return st.button(
            "Run Benchmark",
            disabled=not model_keys,
            width="stretch",
            type="primary",
        )


def _sidebar_configure_button() -> None:
    """Rendered after the run handler so enabled state reflects stored results."""
    has_report = _has_stored_report()
    with st.sidebar:
        if st.button(
            "Configure new run",
            key="sidebar_configure_new_run",
            width="stretch",
            disabled=not has_report,
            help=(
                "Return to document selection and model setup. Keeps your current checkbox choices."
                if has_report
                else "Available after you run a benchmark."
            ),
        ):
            _clear_run_state()
            st.rerun()


# ---------------------------------------------------------------------------
# Pre-run: task context + metric explainers + model table + doc grid
# ---------------------------------------------------------------------------

def _render_pre_run(
    task: str,
    model_keys: list[str],
    sample_size: int,
    config: dict,
    all_docs: list[dict],
) -> None:
    meta = _TASK_META.get(task, {})
    catalog = get_model_catalog(config)

    # Task context card
    with st.container(border=True):
        cols = st.columns([3, 1, 1])
        cols[0].markdown(f"### {meta.get('label', task.title())}")
        cols[0].markdown(meta.get("description", ""))
        cols[1].metric("Dataset", meta.get("dataset", "—"))
        cols[2].metric("Metrics", "  ·  ".join(meta.get("metrics", [])))

    st.divider()

    # Metric explainer cards — one per metric, side by side
    st.markdown("#### What will be measured")
    metric_cols = st.columns(len(meta.get("metrics", [])))
    for i, metric_name in enumerate(meta.get("metrics", [])):
        info = _METRIC_DESCRIPTIONS.get(metric_name, {})
        with metric_cols[i]:
            with st.container(border=True):
                st.markdown(f"**{metric_name}**")
                st.caption(info.get("short", ""))
                st.markdown(
                    f"Range: `{info.get('range', '—')}`  ·  Good score: `{info.get('good', '—')}`"
                )
                with st.expander("How it works"):
                    st.markdown(info.get("detail", ""))

    st.divider()

    # Model comparison table
    if not model_keys:
        st.info("Select at least one model in the sidebar.", icon="👈")
        return

    st.markdown(f"#### {len(model_keys)} model{'s' if len(model_keys) > 1 else ''} selected")
    rows = []
    for key in model_keys:
        m = catalog.get(key, {})
        pricing = m.get("pricing", {})
        rows.append({
            "Model": key,
            "Provider": m.get("provider_type", "—").replace("_", " ").title(),
            "Model ID": m.get("model_id", "—"),
            "Input $/1M": f"${pricing.get('input_per_1m', 0):.3f}",
            "Output $/1M": f"${pricing.get('output_per_1m', 0):.3f}",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(
        "⚠️ Pricing shown are estimates sourced from provider documentation at the time of "
        "writing and may not reflect current rates. Actual costs depend on when and how you "
        "call the API. To update a price, edit `pricing.input_per_1m` / `pricing.output_per_1m` "
        "for the relevant model in `configs/config.yaml`."
    )

    st.divider()

    # Doc selection grid
    if not all_docs:
        st.info(
            "Dataset not found locally. Run `python main.py` to download and prepare.", icon="⚠️"
        )
        return

    truncation_limit: int = (
        config.get("pipeline", {})
        .get("max_document_length_per_task", {})
        .get(task, config.get("pipeline", {}).get("max_document_length", 2000))
    )
    _render_doc_grid(all_docs, sample_size, meta, truncation_limit, model_keys, catalog)


def _render_doc_grid(
    all_docs: list[dict],
    sample_size: int,
    meta: dict,
    truncation_limit: int,
    model_keys: list[str],
    catalog: dict,
) -> None:
    """Render an interactive card grid for selecting which docs to benchmark."""
    n_selected = sum(
        1 for d in all_docs if st.session_state.get(f"_doc_cb_{d['doc_id']}", False)
    )

    # Header + quick-action buttons
    st.markdown("#### Select documents to benchmark")
    col_count, col_all, col_none, col_rand = st.columns([4, 1, 1, 1.4])
    col_count.markdown(
        f"**{n_selected} of {len(all_docs)} docs selected** — "
        "these will be sent to each model on Run"
    )

    if col_all.button("All", use_container_width=True, key="_btn_all"):
        for doc in all_docs:
            st.session_state[f"_doc_cb_{doc['doc_id']}"] = True

    if col_none.button("None", use_container_width=True, key="_btn_none"):
        for doc in all_docs:
            st.session_state[f"_doc_cb_{doc['doc_id']}"] = False

    if col_rand.button(f"Random {sample_size}", use_container_width=True, key="_btn_rand"):
        chosen = {
            d["doc_id"]
            for d in random.sample(all_docs, min(sample_size, len(all_docs)))
        }
        for doc in all_docs:
            st.session_state[f"_doc_cb_{doc['doc_id']}"] = doc["doc_id"] in chosen

    # Toggle: selected docs only vs. all 30
    show_all: bool = st.toggle(
        f"Show all {len(all_docs)} docs",
        key="_show_all_docs",
        help="Enable to browse the full dataset and check individual docs.",
    )

    docs_to_show = all_docs if show_all else [
        d for d in all_docs if st.session_state.get(f"_doc_cb_{d['doc_id']}", False)
    ]

    if not docs_to_show:
        st.info(
            "No docs selected. Use the buttons above or enable **Show all docs** to pick from the full dataset."
        )
        return

    preview_field = meta.get("preview_field", "raw_text")
    ref_field = meta.get("preview_ref_field", "")
    preview_label = meta.get("preview_label", "Source")
    ref_label = meta.get("preview_ref_label", "Reference")

    # Adapt text truncation to card count — fewer cards = more breathing room
    text_limit = 300 if len(docs_to_show) <= 6 else 140

    for row_start in range(0, len(docs_to_show), 3):
        row = docs_to_show[row_start: row_start + 3]
        cols = st.columns(3)
        for col, doc in zip(cols, row):
            doc_id = doc["doc_id"]
            raw = doc.get(preview_field, "")
            ref = doc.get("metadata", {}).get(ref_field, "")

            with col:
                with st.container(border=True):
                    st.checkbox(f"`{doc_id}`", key=f"_doc_cb_{doc_id}")
                    hint = _token_cost_hint(
                        raw,
                        doc.get("source_language", "en"),
                        truncation_limit,
                        model_keys,
                        catalog,
                    )
                    st.caption(hint)
                    st.caption(preview_label)
                    st.markdown(
                        f"<small>{raw[:text_limit]}{'…' if len(raw) > text_limit else ''}</small>",
                        unsafe_allow_html=True,
                    )
                    if ref:
                        st.caption(ref_label)
                        st.markdown(
                            f"<small>{ref[:text_limit]}{'…' if len(ref) > text_limit else ''}</small>",
                            unsafe_allow_html=True,
                        )


# ---------------------------------------------------------------------------
# Benchmark execution — with per-model progress
# ---------------------------------------------------------------------------

def _run_with_progress(
    runner: BenchmarkRunner,
    task: str,
    model_keys: list[str],
    documents: list[DocumentInput],
) -> BenchmarkReport:
    """Run each model and surface real-time per-model progress in the UI."""
    n = len(model_keys)
    n_docs = len(documents)
    metric_names = " · ".join(_TASK_META.get(task, {}).get("metrics", ["metrics"]))

    accumulated: list[ModelBenchmarkResult] = []
    progress = st.progress(0.0, text="Starting benchmark…")
    completed_log = st.empty()      # grows: one line per finished model
    current_step = st.empty()       # replaced each iteration: shows current model

    completed_lines: list[str] = []

    for i, model_key in enumerate(model_keys):
        progress.progress(i / n, text=f"Step {i + 1}/{n} — {model_key}")
        current_step.markdown(
            f"⏳ **{model_key}** — sending {n_docs} doc{'s' if n_docs != 1 else ''} "
            f"to API for {task}…"
        )

        t0 = time.perf_counter()
        single = runner.run(
            task=task,
            model_keys=[model_key],
            sample_size=n_docs,
            documents=documents,
        )
        elapsed = time.perf_counter() - t0

        accumulated.extend(single.results)

        completed_lines.append(
            f"✅ **{model_key}** — {elapsed:.1f}s · scored with {metric_names}"
        )
        completed_log.markdown("\n\n".join(completed_lines))
        current_step.empty()
        progress.progress((i + 1) / n, text=f"Step {i + 1}/{n} — {model_key} complete")

    progress.progress(1.0, text=f"✅ {n} model{'s' if n > 1 else ''} evaluated on {n_docs} docs")
    return BenchmarkReport(
        task=task,
        sample_size=n_docs,
        results=accumulated,
        prompt_version=get_prompt_version(runner.prompts),
    )


# ---------------------------------------------------------------------------
# Results — dataframe, insights, charts, export
# ---------------------------------------------------------------------------

def _to_dataframe(report: BenchmarkReport) -> pd.DataFrame:
    rows = []
    for r in report.results:
        row: dict = {"Model": r.model_key}
        for key, value in r.quality_metrics.items():
            row[_METRIC_DISPLAY.get(key, key.upper())] = round(value, 4)
        cost_per_doc = round(r.total_cost_usd / r.n_docs, 8) if r.n_docs else 0.0
        row.update({
            "Avg In Tokens": round(r.avg_input_tokens),
            "Avg Out Tokens": round(r.avg_output_tokens),
            "Total Cost ($)": r.total_cost_usd,
            "Cost/Doc ($)": cost_per_doc,
            "Avg Latency (ms)": round(r.avg_latency_ms),
            "Docs": r.n_docs,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _quality_context(metric: str, score: float) -> str:
    """Map a metric score to a human-readable verdict."""
    thresholds: dict[str, list[tuple[float, str]]] = {
        "BLEU":      [(0.40, "excellent"), (0.30, "good"), (0.15, "moderate"), (0.0, "low")],
        "ROUGE-L":   [(0.45, "excellent"), (0.30, "good"), (0.15, "moderate"), (0.0, "low")],
        "BERTScore": [(0.80, "excellent"), (0.70, "good"), (0.60, "moderate"), (0.0, "low")],
    }
    for threshold, label in thresholds.get(metric, []):
        if score >= threshold:
            return label
    return "—"


def _render_insights(df: pd.DataFrame) -> None:
    quality_cols = [c for c in df.columns if c in _METRIC_DESCRIPTIONS]
    if not quality_cols or df.empty:
        return

    lines: list[str] = []

    for col in quality_cols:
        best = df.loc[df[col].idxmax()]
        spread = df[col].max() - df[col].min()
        verdict = _quality_context(col, best[col])
        lines.append(
            f"**{col}** — Best: **{best['Model']}** ({best[col]:.4f} — *{verdict}*)"
            + (f"  ·  Spread across models: {spread:.4f}" if len(df) > 1 else "")
        )

    if "Cost/Doc ($)" in df.columns and len(df) > 1:
        cheapest = df.loc[df["Cost/Doc ($)"].idxmin()]
        priciest = df.loc[df["Cost/Doc ($)"].idxmax()]
        ratio = (
            priciest["Cost/Doc ($)"] / cheapest["Cost/Doc ($)"]
            if cheapest["Cost/Doc ($)"] > 0
            else 1
        )
        lines.append(
            f"**Cost** — Cheapest: **{cheapest['Model']}** (${cheapest['Cost/Doc ($)']:.6f}/doc)"
            f"  ·  **{priciest['Model']}** is {ratio:.1f}× pricier per doc"
        )

    if "Avg Latency (ms)" in df.columns and len(df) > 1:
        fastest = df.loc[df["Avg Latency (ms)"].idxmin()]
        slowest = df.loc[df["Avg Latency (ms)"].idxmax()]
        lines.append(
            f"**Latency** — Fastest: **{fastest['Model']}** ({fastest['Avg Latency (ms)']:.0f} ms)"
            f"  ·  Slowest: **{slowest['Model']}** ({slowest['Avg Latency (ms)']:.0f} ms)"
        )

    if len(df) > 1:
        df["_mean_q"] = df[quality_cols].mean(axis=1)
        best_overall = df.loc[df["_mean_q"].idxmax()]
        lines.append(
            f"**Best overall quality** (mean across metrics): "
            f"**{best_overall['Model']}** ({best_overall['_mean_q']:.4f})"
        )
        df.drop(columns=["_mean_q"], inplace=True)

    with st.container(border=True):
        st.markdown("#### Key takeaways")
        for line in lines:
            st.markdown(f"- {line}")
        with st.expander("What do these scores mean?"):
            for col in quality_cols:
                info = _METRIC_DESCRIPTIONS.get(col, {})
                st.markdown(
                    f"**{col}** (`{info.get('range', '—')}`) — "
                    f"{info.get('short', '')}  Good: `{info.get('good', '—')}`."
                )


def _render_metric_bars(df: pd.DataFrame, colors: dict[str, str]) -> None:
    quality_cols = [c for c in df.columns if c in _METRIC_DESCRIPTIONS]
    if not quality_cols:
        return

    st.subheader("Quality Metrics")
    color_scale = alt.Scale(domain=list(colors.keys()), range=list(colors.values()))
    chart_cols = st.columns(len(quality_cols))

    for i, metric in enumerate(quality_cols):
        chart_df = df[["Model", metric]].copy()
        bar = (
            alt.Chart(chart_df)
            .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
            .encode(
                x=alt.X("Model:N", axis=alt.Axis(labelAngle=-20, title=None)),
                y=alt.Y(
                    f"{metric}:Q",
                    scale=alt.Scale(domain=[0, 1]),
                    title="Score",
                    axis=alt.Axis(format=".2f"),
                ),
                color=alt.Color("Model:N", scale=color_scale, legend=None),
                tooltip=[
                    "Model:N",
                    alt.Tooltip(f"{metric}:Q", format=".4f", title=metric),
                ],
            )
        )
        text = bar.mark_text(dy=-6, fontSize=12, fontWeight="bold").encode(
            text=alt.Text(f"{metric}:Q", format=".3f"),
            color=alt.value("#cccccc"),
        )
        chart_cols[i].altair_chart(
            (bar + text).properties(title=metric, height=250),
            use_container_width=True,
        )


def _render_cost_scatter(df: pd.DataFrame, colors: dict[str, str]) -> None:
    if "BERTScore" not in df.columns or df.empty:
        return

    st.subheader("Cost vs. Quality")
    st.caption(
        "Ideal model: **top-left** — highest semantic quality at lowest cost. "
        "Point size encodes avg latency (larger = slower). Hover for details."
    )

    chart_df = df[["Model", "Cost/Doc ($)", "BERTScore", "Avg Latency (ms)"]].copy()
    color_scale = alt.Scale(domain=list(colors.keys()), range=list(colors.values()))

    scatter = (
        alt.Chart(chart_df)
        .mark_circle(opacity=0.9)
        .encode(
            x=alt.X(
                "Cost/Doc ($):Q",
                title="Cost per Doc ($)",
                axis=alt.Axis(format=".5f"),
                scale=alt.Scale(padding=0.4),
            ),
            y=alt.Y(
                "BERTScore:Q",
                title="BERTScore (semantic quality)",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color("Model:N", scale=color_scale),
            size=alt.Size(
                "Avg Latency (ms):Q",
                scale=alt.Scale(range=[150, 500]),
                legend=alt.Legend(title="Avg Latency (ms)"),
            ),
            tooltip=[
                "Model:N",
                alt.Tooltip("BERTScore:Q", format=".4f"),
                alt.Tooltip("Cost/Doc ($):Q", format=".6f"),
                alt.Tooltip("Avg Latency (ms):Q", format=".0f"),
            ],
        )
    )

    labels = (
        alt.Chart(chart_df)
        .mark_text(dy=-16, fontSize=12, fontWeight="bold")
        .encode(
            x="Cost/Doc ($):Q",
            y="BERTScore:Q",
            text="Model:N",
            color=alt.Color("Model:N", scale=color_scale, legend=None),
        )
    )

    st.altair_chart(
        (scatter + labels).properties(height=360).interactive(),
        use_container_width=True,
    )


def _render_export(report: BenchmarkReport, df: pd.DataFrame) -> None:
    st.subheader("Export")
    col_json, col_csv = st.columns(2)
    col_json.download_button(
        "Download JSON",
        json.dumps(report.model_dump(), indent=2, default=str).encode(),
        "benchmark_report.json",
        "application/json",
        use_container_width=True,
    )
    col_csv.download_button(
        "Download CSV",
        df.to_csv(index=False).encode(),
        "benchmark_report.csv",
        "text/csv",
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="LLMevallab", page_icon="🔬", layout="wide")
    st.title("LLMevallab — Model Comparison Dashboard")
    st.caption(
        "Side-by-side quality, cost, and latency benchmark across LLM providers. "
        "Configure settings in the sidebar, then click **Run Benchmark**."
    )

    config, prompts = _load_resources(*_resource_mtimes())
    task, model_keys, sample_size = _sidebar_settings(config)
    data_source = _sidebar_data_source()
    _sidebar_prompt_editor(task, prompts)
    run_clicked = _sidebar_run_button(model_keys)

    # Load documents for the selected data source
    if data_source == _DATA_SOURCE_SAMPLES:
        all_docs = _load_benchmark_samples(task)
    else:
        all_docs = _load_all_docs(config, task)
        if not all_docs:
            st.error(
                "No full dataset found for this task. "
                "Run `python main.py` first to download the full dataset.",
                icon="⚠️",
            )

    # Sync checkbox state — resets on task or sample_size change
    _sync_doc_selection(all_docs, sample_size, task)

    # What the user has currently selected in the grid
    selected_docs = _get_selected_docs(all_docs)

    # ── Handle Run ──────────────────────────────────────────────────────────
    if run_clicked and model_keys and selected_docs:
        cache_key = _cache_key(task, model_keys, [d["doc_id"] for d in selected_docs])
        cached = st.session_state.get("_run_cache", {}).get(cache_key)

        if cached:
            st.session_state["report"] = cached
            st.session_state["report_task"] = task
            st.session_state["_from_cache"] = True
            st.toast("Loaded from cache — same docs & models.", icon="💾")
        else:
            doc_inputs = [DocumentInput(**d) for d in selected_docs]
            runner = BenchmarkRunner(config=config, prompts=prompts)
            report = _run_with_progress(runner, task, model_keys, doc_inputs)
            st.session_state.setdefault("_run_cache", {})[cache_key] = report
            st.session_state["report"] = report
            st.session_state["report_task"] = task
            st.session_state["_from_cache"] = False
            _append_run_history(
                _make_history_entry(
                    report,
                    model_keys,
                    report.prompt_version or get_prompt_version(prompts),
                )
            )

    elif run_clicked and not selected_docs:
        st.warning("No documents selected — check at least one doc in the grid below.", icon="⚠️")

    _sidebar_run_history()

    # Render after run handler so enabled state reflects results stored this pass.
    _sidebar_configure_button()

    # ── Render ───────────────────────────────────────────────────────────────
    report: BenchmarkReport | None = st.session_state.get("report")
    if report and st.session_state.get("report_task") != task:
        report = None  # Discard stale results when task changes

    if report is None:
        _render_pre_run(task, model_keys, sample_size, config, all_docs)
        return

    from_cache = st.session_state.get("_from_cache", False)
    cache_badge = "  ·  💾 from cache" if from_cache else ""
    prompt_badge = (
        f"  ·  prompt v{report.prompt_version}" if report.prompt_version else ""
    )
    selected_ids = _selected_doc_ids(all_docs)

    st.subheader(
        f"Results — task: {report.task}  ·  {report.sample_size} docs"
        f"{prompt_badge}{cache_badge}"
    )
    with st.expander(
        f"Change document selection & re-run ({len(selected_ids)} docs selected)",
        expanded=False,
    ):
        st.caption(
            "Pick different documents below, then click **Run Benchmark** in the sidebar. "
            "Results update in place; identical task/model/doc combos load from cache."
        )
        truncation_limit: int = (
            config.get("pipeline", {})
            .get("max_document_length_per_task", {})
            .get(task, config.get("pipeline", {}).get("max_document_length", 2000))
        )
        _render_doc_grid(
            all_docs,
            sample_size,
            _TASK_META.get(task, {}),
            truncation_limit,
            model_keys,
            get_model_catalog(config),
        )

    df = _to_dataframe(report)
    colors = _model_colors([r.model_key for r in report.results])

    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    _render_insights(df)

    st.divider()
    _render_metric_bars(df, colors)

    st.divider()
    _render_cost_scatter(df, colors)

    st.divider()
    _render_export(report, df)


if __name__ == "__main__":
    main()
