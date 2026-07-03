from unittest.mock import patch

from src.core.config import load_config
from src.core.models import DocumentInput, PipelineResult, SummaryResult
from src.evaluations.evaluator import Evaluator
from src.pipeline.orchestrator import PipelineTask


def test_run_on_results_persist_report_calls_save_report():
    """Regression: persist_report kwarg must not shadow the save_report function."""
    config = load_config()
    evaluator = Evaluator(config)
    pipeline_result = PipelineResult(
        document=DocumentInput(doc_id="d1", source_language="en", raw_text="Article body."),
        summary=SummaryResult(doc_id="d1", summary="Short summary.", model_used="test-model"),
    )

    with patch("src.evaluations.evaluator.load_ground_truth", return_value={"d1": "Reference summary."}), patch(
        "src.evaluations.evaluator.MetricsRunner"
    ) as mock_runner_cls, patch("src.evaluations.evaluator.save_report") as mock_save:
        mock_runner_cls.return_value.run_all.return_value = {"rouge": []}

        evaluator.run_on_results(
            pipeline_results=[pipeline_result],
            ground_truth_path="data/processed/cnn_dailymail/cnn_dailymail_20docs.json",
            task=PipelineTask.SUMMARISATION,
            persist_report=True,
        )

        mock_save.assert_called_once()
