import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.core.config import load_config
from src.core.models import (
    DocumentInput,
    ExtractionResult,
    PipelineResult,
    SummaryResult,
    TranslationResult,
)
from src.core.pricing import TokenUsage
from src.evaluations.benchmark import BenchmarkRunner, save_benchmark_report
from src.pipeline.orchestrator import PipelineTask, load_prompts


class StubProcessor:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def extract(self, document: DocumentInput) -> ExtractionResult:
        return ExtractionResult(
            doc_id=document.doc_id,
            entities=["stub"],
            token_usage=TokenUsage(input_tokens=50, output_tokens=10),
            cost_usd=0.001,
            model_used=self.model_name,
            processing_time_ms=20.0,
        )

    def translate(self, document: DocumentInput, target_language: str = "en") -> TranslationResult:
        return TranslationResult(
            doc_id=document.doc_id,
            source_language=document.source_language,
            target_language=target_language,
            original_text=document.raw_text,
            translated_text="The parliament met.",
            token_usage=TokenUsage(input_tokens=80, output_tokens=20),
            cost_usd=0.002,
            model_used=self.model_name,
            processing_time_ms=30.0,
        )

    def summarise(self, document: DocumentInput) -> SummaryResult:
        return SummaryResult(
            doc_id=document.doc_id,
            summary="Stub summary",
            key_points=["point"],
            token_usage=TokenUsage(input_tokens=60, output_tokens=15),
            cost_usd=0.0015,
            model_used=self.model_name,
            processing_time_ms=25.0,
        )


def test_benchmark_runner_e2e_two_mocked_models(tmp_path):
    config = load_config()
    prompts = load_prompts()
    runner = BenchmarkRunner(config=config, prompts=prompts)

    documents = [
        DocumentInput(
            doc_id="doc-1",
            source_language="de",
            raw_text="Das Parlament tagte.",
            metadata={"reference_translation": "The parliament met."},
        ),
        DocumentInput(
            doc_id="doc-2",
            source_language="de",
            raw_text="Die Abstimmung war einstimmig.",
            metadata={"reference_translation": "The vote was unanimous."},
        ),
    ]

    def fake_build_processor(model_key, config, prompts):
        catalog = config["models"]["catalog"]
        return StubProcessor(catalog[model_key]["model_id"])

    class FakeOrchestrator:
        def __init__(self, processor, config, task, prompt_version=None):
            self.processor = processor

        def run(self, docs):
            return [
                PipelineResult(
                    document=doc,
                    extraction=self.processor.extract(doc),
                    translation=self.processor.translate(doc),
                    total_processing_time_ms=50.0,
                )
                for doc in docs
            ]

    fake_eval_aggregate = {
        "bleu": {"mean": 0.55, "min": 0.5, "max": 0.6, "std": 0.05, "n_docs": 2},
        "bertscore": {"mean": 0.91, "min": 0.9, "max": 0.92, "std": 0.01, "n_docs": 2},
    }

    with patch("src.evaluations.benchmark.load_documents_for_task", return_value=documents), patch(
        "src.evaluations.benchmark.build_processor",
        side_effect=fake_build_processor,
    ), patch(
        "src.evaluations.benchmark.PipelineOrchestrator",
        FakeOrchestrator,
    ), patch(
        "src.evaluations.benchmark.Evaluator.run_on_results",
        side_effect=lambda **kwargs: MagicMock(
            model_used=kwargs["pipeline_results"][0].translation.model_used,
            aggregate=fake_eval_aggregate,
        ),
    ):
        report = runner.run(
            task=PipelineTask.TRANSLATION,
            model_keys=["gemini-2.5-flash", "claude-sonnet-4-6"],
            sample_size=2,
        )

    assert report.sample_size == 2
    assert len(report.results) == 2
    for result in report.results:
        assert result.quality_metrics["bleu"] == 0.55
        assert result.avg_input_tokens > 0
        assert result.total_cost_usd > 0
        assert result.n_docs == 2

    json_path, csv_path = save_benchmark_report(report, tmp_path)
    assert json_path.exists()
    assert csv_path.exists()

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["task"] == "translation"
    assert len(loaded["results"]) == 2

    df = pd.read_csv(csv_path)
    assert len(df) == 2
    assert set(df["model_key"]) == {"gemini-2.5-flash", "claude-sonnet-4-6"}
