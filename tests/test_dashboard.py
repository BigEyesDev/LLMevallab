"""Tests for app/dashboard.py — smoke import and pure-function unit tests."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.core.models import BenchmarkReport, DocumentInput, ModelBenchmarkResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc_input(doc_id: str = "d1") -> DocumentInput:
    return DocumentInput(doc_id=doc_id, source_language="en", raw_text="Some text.")


def _make_report(task: str = "summarisation") -> BenchmarkReport:
    return BenchmarkReport(
        task=task,
        sample_size=5,
        results=[
            ModelBenchmarkResult(
                model_key="test-model",
                model_id="test/model-id",
                quality_metrics={"rouge": 0.42, "bertscore": 0.75},
                avg_input_tokens=300.0,
                avg_output_tokens=80.0,
                total_cost_usd=0.00012,
                avg_latency_ms=1234.5,
                n_docs=5,
            )
        ],
    )


def _make_single_model_report(model_key: str, n_docs: int = 3) -> BenchmarkReport:
    return BenchmarkReport(
        task="summarisation",
        sample_size=n_docs,
        results=[
            ModelBenchmarkResult(
                model_key=model_key,
                model_id=f"org/{model_key}",
                quality_metrics={"rouge": 0.4, "bertscore": 0.7},
                n_docs=n_docs,
            )
        ],
    )


# ---------------------------------------------------------------------------
# Smoke: module importable without a running Streamlit server
# ---------------------------------------------------------------------------

def test_dashboard_importable():
    """Importing dashboard must not raise (catches missing deps, syntax errors)."""
    import app.dashboard  # noqa: F401


# ---------------------------------------------------------------------------
# Unit: _to_dataframe — pure function, no Streamlit dependency
# ---------------------------------------------------------------------------

def test_to_dataframe_columns():
    from app.dashboard import _to_dataframe

    df = _to_dataframe(_make_report())

    assert isinstance(df, pd.DataFrame)
    assert "Model" in df.columns
    assert "ROUGE-L" in df.columns
    assert "BERTScore" in df.columns
    assert "Total Cost ($)" in df.columns
    assert "Cost/Doc ($)" in df.columns
    assert "Avg Latency (ms)" in df.columns


def test_to_dataframe_model_name():
    from app.dashboard import _to_dataframe

    assert _to_dataframe(_make_report()).iloc[0]["Model"] == "test-model"


def test_to_dataframe_cost_per_doc():
    from app.dashboard import _to_dataframe

    df = _to_dataframe(_make_report())
    assert df.iloc[0]["Cost/Doc ($)"] == pytest.approx(round(0.00012 / 5, 8))


def test_to_dataframe_metric_rounding():
    from app.dashboard import _to_dataframe

    df = _to_dataframe(_make_report())
    assert df.iloc[0]["ROUGE-L"] == pytest.approx(0.42, abs=1e-4)
    assert df.iloc[0]["BERTScore"] == pytest.approx(0.75, abs=1e-4)


def test_to_dataframe_multiple_models():
    from app.dashboard import _to_dataframe

    results = [
        ModelBenchmarkResult(
            model_key=f"model-{i}",
            model_id=f"org/model-{i}",
            quality_metrics={"bertscore": 0.5 + i * 0.1},
            n_docs=3,
        )
        for i in range(3)
    ]
    df = _to_dataframe(BenchmarkReport(task="translation", sample_size=3, results=results))
    assert len(df) == 3
    assert list(df["Model"]) == ["model-0", "model-1", "model-2"]


# ---------------------------------------------------------------------------
# Unit: _quality_context — score-to-verdict mapping
# ---------------------------------------------------------------------------

def test_quality_context_bertscore_excellent():
    from app.dashboard import _quality_context
    assert _quality_context("BERTScore", 0.85) == "excellent"


def test_quality_context_bertscore_good():
    from app.dashboard import _quality_context
    assert _quality_context("BERTScore", 0.72) == "good"


def test_quality_context_rouge_moderate():
    from app.dashboard import _quality_context
    assert _quality_context("ROUGE-L", 0.20) == "moderate"


def test_quality_context_bleu_low():
    from app.dashboard import _quality_context
    assert _quality_context("BLEU", 0.05) == "low"


# ---------------------------------------------------------------------------
# Unit: _model_colors — deterministic, consistent assignment
# ---------------------------------------------------------------------------

def test_model_colors_unique_for_distinct_models():
    from app.dashboard import _model_colors
    assert len(set(_model_colors(["a", "b", "c"]).values())) == 3


def test_model_colors_consistent_across_calls():
    from app.dashboard import _model_colors
    assert _model_colors(["x", "y"]) == _model_colors(["x", "y"])


def test_model_colors_wraps_palette():
    from app.dashboard import _model_colors, _PALETTE
    keys = [f"m{i}" for i in range(len(_PALETTE) + 2)]
    colors = _model_colors(keys)
    assert colors[keys[0]] == colors[keys[len(_PALETTE)]]


# ---------------------------------------------------------------------------
# Unit: _cache_key — deterministic regardless of order
# ---------------------------------------------------------------------------

def test_cache_key_order_independent():
    from app.dashboard import _cache_key
    assert (
        _cache_key("summarisation", ["b", "a"], ["d2", "d1"])
        == _cache_key("summarisation", ["a", "b"], ["d1", "d2"])
    )


def test_cache_key_differs_by_task():
    from app.dashboard import _cache_key
    assert (
        _cache_key("translation", ["m"], ["d1"])
        != _cache_key("summarisation", ["m"], ["d1"])
    )


# ---------------------------------------------------------------------------
# Unit: _load_benchmark_samples — correct loader called, graceful on missing
# ---------------------------------------------------------------------------

def test_load_benchmark_samples_returns_dicts_on_success(tmp_path):
    """When sample files are present, _load_benchmark_samples returns list[dict]."""
    import json
    from app.dashboard import _load_benchmark_samples

    docs = [
        {
            "doc_id": f"europarl_de-en_{i:04d}",
            "source_language": "de",
            "raw_text": f"Text {i}",
            "source": "europarl",
            "metadata": {"reference_translation": f"Ref {i}"},
        }
        for i in range(3)
    ]
    (tmp_path / "translation_de_en.json").write_text(json.dumps(docs), encoding="utf-8")

    with patch("app.dashboard.BenchmarkSampleLoader") as MockLoader:
        from src.core.models import DocumentInput

        MockLoader.return_value.load.return_value = [DocumentInput(**d) for d in docs]
        # Clear Streamlit's cache so the mock is actually called
        _load_benchmark_samples.clear()
        result = _load_benchmark_samples("translation")

    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(d, dict) for d in result)
    assert result[0]["doc_id"] == "europarl_de-en_0000"


def test_load_benchmark_samples_returns_empty_on_missing_file():
    """When sample files are absent, _load_benchmark_samples returns []."""
    from app.dashboard import _load_benchmark_samples

    with patch("app.dashboard.BenchmarkSampleLoader") as MockLoader:
        MockLoader.return_value.load.side_effect = FileNotFoundError("not found")
        _load_benchmark_samples.clear()
        result = _load_benchmark_samples("translation")

    assert result == []


def test_data_source_constants_defined():
    from app.dashboard import _DATA_SOURCE_SAMPLES, _DATA_SOURCE_FULL

    assert _DATA_SOURCE_SAMPLES == "Benchmark samples"
    assert _DATA_SOURCE_FULL == "Full dataset"


def test_clear_run_state_removes_report_keys():
    from app.dashboard import _clear_run_state

    fake_state = {
        "report": "dummy",
        "report_task": "summarisation",
        "_from_cache": True,
        "_doc_cb_d1": True,
    }
    with patch("app.dashboard.st.session_state", fake_state):
        _clear_run_state()

    assert "report" not in fake_state
    assert "report_task" not in fake_state
    assert "_from_cache" not in fake_state
    assert fake_state["_doc_cb_d1"] is True  # doc selection preserved


def test_has_stored_report():
    from app.dashboard import _has_stored_report

    with patch("app.dashboard.st.session_state", {}):
        assert not _has_stored_report()

    with patch("app.dashboard.st.session_state", {"report": object()}):
        assert _has_stored_report()


def test_selected_doc_ids_returns_checked_docs():
    from app.dashboard import _selected_doc_ids

    docs = [
        {"doc_id": "d1", "raw_text": "a"},
        {"doc_id": "d2", "raw_text": "b"},
    ]
    fake_state = {"_doc_cb_d1": True, "_doc_cb_d2": False}

    with patch("app.dashboard.st.session_state", fake_state):
        assert _selected_doc_ids(docs) == ["d1"]


# ---------------------------------------------------------------------------
# Unit: _get_selected_docs — reads from session_state
# ---------------------------------------------------------------------------

def test_get_selected_docs_filters_by_checkbox_state():
    from app.dashboard import _get_selected_docs

    docs = [
        {"doc_id": "d1", "raw_text": "a"},
        {"doc_id": "d2", "raw_text": "b"},
        {"doc_id": "d3", "raw_text": "c"},
    ]
    fake_state = {"_doc_cb_d1": True, "_doc_cb_d2": False, "_doc_cb_d3": True}

    with patch("app.dashboard.st.session_state", fake_state):
        selected = _get_selected_docs(docs)

    assert [d["doc_id"] for d in selected] == ["d1", "d3"]


# ---------------------------------------------------------------------------
# Unit: _run_with_progress — mock runner and Streamlit widgets
# ---------------------------------------------------------------------------

def _mock_progress():
    p = MagicMock()
    p.progress = MagicMock()
    return p


def _run_progress_patched(runner, task, model_keys, documents):
    """Run _run_with_progress with all Streamlit calls mocked."""
    from app.dashboard import _run_with_progress

    mock_progress = MagicMock()
    mock_empty = MagicMock()

    with (
        patch("app.dashboard.st.progress", return_value=mock_progress),
        patch("app.dashboard.st.empty", return_value=mock_empty),
        patch("app.dashboard.time.perf_counter", return_value=0.0),
    ):
        return _run_with_progress(runner, task, model_keys, documents)


def test_run_with_progress_calls_runner_per_model():
    model_keys = ["model-a", "model-b"]
    documents = [_make_doc_input("d1"), _make_doc_input("d2"), _make_doc_input("d3")]

    runner = MagicMock()
    runner.run.side_effect = [
        _make_single_model_report("model-a", n_docs=3),
        _make_single_model_report("model-b", n_docs=3),
    ]

    report = _run_progress_patched(runner, "summarisation", model_keys, documents)

    assert runner.run.call_count == 2
    runner.run.assert_any_call(
        task="summarisation", model_keys=["model-a"], sample_size=3, documents=documents
    )
    runner.run.assert_any_call(
        task="summarisation", model_keys=["model-b"], sample_size=3, documents=documents
    )
    assert len(report.results) == 2
    assert report.task == "summarisation"
    assert report.sample_size == 3


def test_run_with_progress_combines_results():
    model_keys = ["alpha", "beta", "gamma"]
    documents = [_make_doc_input(f"d{i}") for i in range(3)]

    runner = MagicMock()
    runner.run.side_effect = [_make_single_model_report(k, n_docs=3) for k in model_keys]

    report = _run_progress_patched(runner, "summarisation", model_keys, documents)

    assert len(report.results) == 3
    assert [r.model_key for r in report.results] == model_keys
