import json

from src.core.models import (
    BenchmarkReport,
    DocumentInput,
    ModelBenchmarkResult,
    PipelineResult,
    TranslationResult,
)
from src.core.pricing import TokenUsage


def test_result_fields_default_without_token_usage():
    result = TranslationResult(
        doc_id="test_001",
        source_language="de",
        original_text="Hallo",
        translated_text="Hello",
    )
    assert result.token_usage is None
    assert result.cost_usd == 0.0


def test_pipeline_result_round_trip_with_token_usage():
    doc = DocumentInput(
        doc_id="europarl_de-en_0000",
        source_language="de",
        raw_text="Guten Tag",
    )
    translation = TranslationResult(
        doc_id=doc.doc_id,
        source_language="de",
        original_text=doc.raw_text,
        translated_text="Good day",
        token_usage=TokenUsage(input_tokens=120, output_tokens=30),
        cost_usd=0.000018,
    )
    pipeline_result = PipelineResult(document=doc, translation=translation)

    restored = PipelineResult.model_validate(json.loads(pipeline_result.model_dump_json()))
    assert restored.translation.token_usage.input_tokens == 120
    assert restored.translation.token_usage.output_tokens == 30
    assert restored.translation.token_usage.total_tokens == 150
    assert restored.translation.cost_usd == 0.000018


def test_pipeline_result_carries_prompt_version():
    doc = DocumentInput(doc_id="d1", source_language="de", raw_text="Hallo")
    result = PipelineResult(document=doc, prompt_version="4")
    restored = PipelineResult.model_validate(json.loads(result.model_dump_json()))
    assert restored.prompt_version == "4"


def test_benchmark_report_round_trip():
    report = BenchmarkReport(
        task="translation",
        sample_size=20,
        results=[
            ModelBenchmarkResult(
                model_key="gemini-2.5-flash",
                model_id="gemini-2.5-flash",
                quality_metrics={"bleu": 0.42, "bertscore": 0.88},
                avg_input_tokens=500.0,
                avg_output_tokens=150.0,
                total_cost_usd=0.05,
                avg_latency_ms=1200.0,
                n_docs=20,
            ),
            ModelBenchmarkResult(
                model_key="claude-sonnet-4-6",
                model_id="claude-sonnet-4-6",
                quality_metrics={"bleu": 0.45, "bertscore": 0.90},
                avg_input_tokens=480.0,
                avg_output_tokens=140.0,
                total_cost_usd=0.80,
                avg_latency_ms=1500.0,
                n_docs=20,
            ),
        ],
    )

    restored = BenchmarkReport.model_validate(json.loads(report.model_dump_json()))
    assert len(restored.results) == 2
    assert restored.results[0].model_key == "gemini-2.5-flash"
    assert restored.results[1].quality_metrics["bleu"] == 0.45
