"""Tests for src/evaluations/metric_registry.py."""

import pandas as pd

from src.core.config import load_config
from src.evaluations.metric_registry import (
    display_name,
    get_task_metric_display_names,
    get_task_metric_keys,
    metric_info_for_display,
    normalize_metric_key,
    primary_scatter_metric,
    quality_columns_in_dataframe,
    quality_context,
)


def test_normalize_metric_key_rouge_alias():
    assert normalize_metric_key("rouge") == "rouge_l"


def test_display_name_known_and_unknown():
    assert display_name("comet") == "COMET"
    assert display_name("llm_judge") == "LLM Judge"
    assert display_name("future_metric") == "Future Metric"


def test_get_task_metrics_from_config():
    config = load_config()
    translation = get_task_metric_display_names(config, "translation")
    assert "BLEU" in translation
    assert "COMET" in translation

    summarisation = get_task_metric_display_names(config, "summarisation")
    assert "ROUGE-L" in summarisation
    assert "LLM Judge" in summarisation


def test_quality_context_comet():
    assert quality_context("COMET", 0.85) == "good"


def test_quality_columns_excludes_operational():
    df = pd.DataFrame({
        "Model": ["a"],
        "COMET": [0.9],
        "Cost/Doc ($)": [0.001],
    })
    assert quality_columns_in_dataframe(df) == ["COMET"]


def test_primary_scatter_metric_prefers_bertscore():
    config = load_config()
    df = pd.DataFrame({
        "Model": ["a", "b"],
        "BERTScore": [0.8, 0.7],
        "COMET": [0.9, 0.85],
        "Cost/Doc ($)": [0.001, 0.002],
    })
    assert primary_scatter_metric(config, "translation", df) == "BERTScore"


def test_metric_info_fallback_for_unknown():
    info = metric_info_for_display("Future Metric")
    assert info["range"] == "0 – 1"
    assert "Higher is better" in info["good"]
