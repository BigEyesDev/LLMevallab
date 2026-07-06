from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.core.config import load_config
from src.evaluations.evaluator import get_task_metrics
from src.evaluations.metrics import COMETMetric, LLMJudgeMetric, MetricInput, MetricsRunner
from src.pipeline.orchestrator import PipelineTask


def _mock_comet_output(scores: list[float], system_score: float):
    return SimpleNamespace(scores=scores, system_score=system_score)


def test_comet_metric_scores_documents_and_corpus():
    mock_model = MagicMock()
    mock_model.predict.return_value = _mock_comet_output([0.85, 0.72], 0.785)

    metric = COMETMetric(model=mock_model)
    inputs = [
        MetricInput(
            doc_id="d1",
            hypothesis="The fire could be stopped",
            reference="They were able to control the fire.",
            source="Dem Feuer konnte Einhalt geboten werden",
        ),
        MetricInput(
            doc_id="d2",
            hypothesis="Schools were open",
            reference="Schools and kindergartens opened",
            source="Schulen und Kindergärten wurden eröffnet.",
        ),
    ]

    scores = metric.score(inputs)

    assert len(scores) == 3
    assert scores[0].doc_id == "d1"
    assert scores[0].metric_name == "comet"
    assert scores[0].score == 0.85
    assert scores[1].score == 0.72
    assert scores[2].doc_id == "__corpus__"
    assert scores[2].metric_name == "comet_corpus"
    assert scores[2].score == 0.785

    call_data = mock_model.predict.call_args[0][0]
    assert call_data[0]["src"] == inputs[0].source
    assert call_data[0]["mt"] == inputs[0].hypothesis
    assert call_data[0]["ref"] == inputs[0].reference


def test_comet_metric_lazy_import_raises_without_package():
    metric = COMETMetric()
    with patch.dict("sys.modules", {"comet": None}), pytest.raises(ImportError, match="unbabel-comet"):
        metric._load_model()


def test_metrics_runner_includes_comet():
    mock_model = MagicMock()
    mock_model.predict.return_value = _mock_comet_output([0.9], 0.9)

    runner = MetricsRunner(metrics=["comet"], comet_model="Unbabel/wmt22-comet-da")
    runner._metrics[0]._model = mock_model

    inputs = [
        MetricInput(
            doc_id="d1",
            hypothesis="Hello",
            reference="Hi",
            source="Hallo",
        )
    ]
    result = runner.run_all(inputs)

    assert "comet" in result
    assert result["comet"][0].score == 0.9


def test_get_task_metrics_reads_config():
    config = load_config()
    metrics = get_task_metrics(config, PipelineTask.TRANSLATION)
    assert "comet" in metrics
    assert "bleu" in metrics


def test_config_translation_metrics_include_comet():
    config = load_config()
    translation_metrics = config["evaluation"]["metrics"]["translation"]
    assert "comet" in translation_metrics


def test_llm_judge_metric_scores_with_mock_client():
    mock_client = MagicMock()
    mock_client.evaluate.return_value = (
        {"faithfulness": 5, "completeness": 4, "coherence": 3},
        120.5,
        {
            "judge_model": "gpt-4o-mini",
            "judge_model_key": "gpt-4o-mini",
            "latency_ms": 120.5,
            "input_tokens": 500,
            "output_tokens": 50,
            "cost_usd": 0.0001,
            "reasoning": "Good summary.",
        },
    )

    metric = LLMJudgeMetric(config={}, judge_model_key="gpt-4o-mini", client=mock_client)
    inputs = [
        MetricInput(
            doc_id="d1",
            hypothesis="A short summary.",
            reference="Reference summary.",
            source="Long article text about an event.",
        )
    ]

    scores = metric.score(inputs)

    assert len(scores) == 1
    assert scores[0].metric_name == "llm_judge"
    assert scores[0].score == pytest.approx(0.75)  # mean of 1.0, 0.75, 0.5
    assert scores[0].metadata["faithfulness"] == 5
    assert scores[0].metadata["latency_ms"] == 120.5
    assert scores[0].metadata["cost_usd"] == 0.0001
    mock_client.evaluate.assert_called_once_with(
        "Long article text about an event.",
        "A short summary.",
    )


def test_llm_judge_skips_missing_source():
    mock_client = MagicMock()
    metric = LLMJudgeMetric(config={}, client=mock_client)
    inputs = [
        MetricInput(doc_id="d1", hypothesis="Summary.", reference="Ref.", source=""),
    ]

    scores = metric.score(inputs)

    assert scores == []
    mock_client.evaluate.assert_not_called()


def test_metrics_runner_includes_llm_judge():
    mock_client = MagicMock()
    mock_client.evaluate.return_value = (
        {"faithfulness": 4, "completeness": 4, "coherence": 4},
        50.0,
        {"judge_model": "gpt-4o-mini", "judge_model_key": "gpt-4o-mini",
         "latency_ms": 50.0, "input_tokens": 100, "output_tokens": 20,
         "cost_usd": 0.0, "reasoning": ""},
    )

    runner = MetricsRunner(
        metrics=["llm_judge"],
        config={"models": {"catalog": {}}},
        judge_model_key="gpt-4o-mini",
    )
    runner._metrics[0]._client = mock_client

    inputs = [
        MetricInput(
            doc_id="d1",
            hypothesis="Summary.",
            reference="Ref.",
            source="Article body.",
        )
    ]
    result = runner.run_all(inputs)

    assert "llm_judge" in result
    assert result["llm_judge"][0].score == pytest.approx(0.75)


def test_config_summarisation_metrics_include_llm_judge():
    config = load_config()
    summarisation_metrics = config["evaluation"]["metrics"]["summarisation"]
    assert "llm_judge" in summarisation_metrics
