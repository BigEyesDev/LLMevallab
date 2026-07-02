import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.core.config import load_config
from src.core.models import (
    BenchmarkReport,
    DocumentInput,
    EvaluationReport,
    ExtractionResult,
    PipelineResult,
    TranslationResult,
)
from src.core.pricing import TokenUsage
from src.evaluations.benchmark import (
    BenchmarkRunner,
    aggregate_pipeline_results,
    load_documents_for_task,
    parse_models,
    print_benchmark_table,
    save_benchmark_report,
)
from src.pipeline.orchestrator import PipelineTask, load_prompts


@pytest.fixture
def sample_document():
    return DocumentInput(
        doc_id="doc-1",
        source_language="de",
        raw_text="Das Parlament tagte.",
        source="test",
        metadata={"reference_translation": "The parliament met."},
    )


@pytest.fixture
def config_and_prompts():
    return load_config(), load_prompts()


def test_parse_models_splits_comma_separated_keys():
    assert parse_models("gemini-2.5-flash, claude-sonnet-4-6") == [
        "gemini-2.5-flash",
        "claude-sonnet-4-6",
    ]


def test_cli_unknown_model_raises_before_run(config_and_prompts):
    config, prompts = config_and_prompts
    runner = BenchmarkRunner(config=config, prompts=prompts)
    with pytest.raises(ValueError, match="Unknown model key"):
        runner.run(
            task=PipelineTask.TRANSLATION,
            model_keys=["not-a-real-model"],
            sample_size=1,
        )


def test_benchmark_core_loop_one_result_per_model_and_same_docs(config_and_prompts, sample_document):
    config, prompts = config_and_prompts
    runner = BenchmarkRunner(config=config, prompts=prompts)
    docs_seen: list[list[str]] = []

    fake_pipeline = [
        PipelineResult(
            document=sample_document,
            translation=TranslationResult(
                doc_id="doc-1",
                source_language="de",
                target_language="en",
                original_text=sample_document.raw_text,
                translated_text="The parliament met.",
                model_used="fake-model",
                processing_time_ms=100.0,
                token_usage=TokenUsage(input_tokens=10, output_tokens=5),
                cost_usd=0.001,
            ),
            total_processing_time_ms=100.0,
        )
    ]
    fake_eval = EvaluationReport(
        model_used="fake-model",
        aggregate={"bleu": {"mean": 0.5, "min": 0.5, "max": 0.5, "std": 0.0, "n_docs": 1}},
    )

    class FakeOrchestrator:
        def __init__(self, processor, config, task):
            pass

        def run(self, documents):
            docs_seen.append([d.doc_id for d in documents])
            return fake_pipeline

    with patch(
        "src.evaluations.benchmark.load_documents_for_task",
        return_value=[sample_document],
    ), patch(
        "src.evaluations.benchmark.build_processor",
        return_value=MagicMock(model_name="fake-model"),
    ), patch(
        "src.evaluations.benchmark.PipelineOrchestrator",
        FakeOrchestrator,
    ), patch(
        "src.evaluations.benchmark.Evaluator.run_on_results",
        return_value=fake_eval,
    ):
        report = runner.run(
            task=PipelineTask.TRANSLATION,
            model_keys=["gemini-2.5-flash", "claude-sonnet-4-6"],
            sample_size=1,
        )

    assert len(report.results) == 2
    assert {r.model_key for r in report.results} == {"gemini-2.5-flash", "claude-sonnet-4-6"}
    assert len(docs_seen) == 2
    assert docs_seen[0] == docs_seen[1] == ["doc-1"]


def test_aggregate_pipeline_results_computes_means():
    doc = DocumentInput(
        doc_id="d1",
        source_language="de",
        raw_text="text",
        metadata={},
    )
    results = [
        PipelineResult(
            document=doc,
            extraction=ExtractionResult(
                doc_id="d1",
                token_usage=TokenUsage(input_tokens=100, output_tokens=20),
                cost_usd=0.01,
                processing_time_ms=50.0,
            ),
            translation=TranslationResult(
                doc_id="d1",
                source_language="de",
                target_language="en",
                original_text="text",
                translated_text="text en",
                token_usage=TokenUsage(input_tokens=200, output_tokens=40),
                cost_usd=0.02,
                processing_time_ms=60.0,
            ),
            total_processing_time_ms=110.0,
        ),
        PipelineResult(
            document=doc,
            translation=TranslationResult(
                doc_id="d1",
                source_language="de",
                target_language="en",
                original_text="text",
                translated_text="text en 2",
                token_usage=None,
                cost_usd=0.0,
                processing_time_ms=70.0,
            ),
            total_processing_time_ms=70.0,
        ),
    ]
    eval_report = EvaluationReport(
        model_used="m",
        aggregate={"bleu": {"mean": 0.42, "min": 0.4, "max": 0.44, "std": 0.01, "n_docs": 2}},
    )

    aggregated = aggregate_pipeline_results("gemini-2.5-flash", "gemini-2.5-flash", results, eval_report)

    assert aggregated.quality_metrics["bleu"] == 0.42
    assert aggregated.avg_input_tokens == pytest.approx(150.0)
    assert aggregated.avg_output_tokens == pytest.approx(30.0)
    assert aggregated.total_cost_usd == pytest.approx(0.03)
    assert aggregated.avg_latency_ms == pytest.approx(90.0)
    assert aggregated.n_docs == 2


def test_save_benchmark_report_round_trips_json_and_csv(tmp_path):
    report = BenchmarkReport(
        task="translation",
        sample_size=2,
        results=[
            {
                "model_key": "gemini-2.5-flash",
                "model_id": "gemini-2.5-flash",
                "quality_metrics": {"bleu": 0.4, "bertscore": 0.9},
                "avg_input_tokens": 100.0,
                "avg_output_tokens": 50.0,
                "total_cost_usd": 0.02,
                "avg_latency_ms": 1200.0,
                "n_docs": 2,
            }
        ],
    )

    json_path, csv_path = save_benchmark_report(report, tmp_path)

    loaded = BenchmarkReport.model_validate(json.loads(json_path.read_text(encoding="utf-8")))
    assert loaded.task == "translation"
    assert loaded.results[0].model_key == "gemini-2.5-flash"

    df = pd.read_csv(csv_path)
    assert len(df) == 1
    assert "model_key" in df.columns
    assert "quality_bleu" in df.columns
    assert "total_cost_usd" in df.columns
    assert df.iloc[0]["model_key"] == "gemini-2.5-flash"


def test_print_benchmark_table_includes_model_names(capsys):
    report = BenchmarkReport(
        task="translation",
        sample_size=1,
        results=[
            {
                "model_key": "gemini-2.5-flash",
                "model_id": "gemini-2.5-flash",
                "quality_metrics": {"bleu": 0.5},
                "avg_input_tokens": 10.0,
                "avg_output_tokens": 5.0,
                "total_cost_usd": 0.001,
                "avg_latency_ms": 100.0,
                "n_docs": 1,
            },
            {
                "model_key": "claude-sonnet-4-6",
                "model_id": "claude-sonnet-4-6",
                "quality_metrics": {"bleu": 0.6},
                "avg_input_tokens": 12.0,
                "avg_output_tokens": 6.0,
                "total_cost_usd": 0.002,
                "avg_latency_ms": 110.0,
                "n_docs": 1,
            },
        ],
    )

    print_benchmark_table(report)
    output = capsys.readouterr().out
    assert "gemini-2.5-fla" in output
    assert "claude-sonnet-" in output


def test_load_documents_for_task_slices_sample(config_and_prompts):
    config, _ = config_and_prompts
    docs = [
        DocumentInput(doc_id=f"d{i}", source_language="de", raw_text="x", metadata={})
        for i in range(5)
    ]
    with patch(
        "src.evaluations.benchmark.EuroParlDataLoader.load_from_disk",
        return_value=docs,
    ):
        loaded = load_documents_for_task(PipelineTask.TRANSLATION, config, sample_size=3)
    assert len(loaded) == 3
    assert [d.doc_id for d in loaded] == ["d0", "d1", "d2"]
