"""Tests for parallel document processing and skip-extraction."""

import threading
from unittest.mock import MagicMock

import pytest

from src.core.models import DocumentInput, ExtractionResult, SummaryResult, TranslationResult
from src.pipeline.orchestrator import PipelineOrchestrator, PipelineTask


@pytest.fixture
def base_config(tmp_path):
    return {
        "pipeline": {
            "target_language": "en",
            "max_concurrent_documents": 3,
            "skip_extraction": False,
            "max_document_length_per_task": {"summarisation": 8000},
        },
        "paths": {"outputs": str(tmp_path)},
        "models": {"catalog": {"stub-model": {"model_id": "stub", "provider_type": "gemini"}}},
    }


def _summarisation_processor():
    proc = MagicMock()
    proc.model_name = "stub-model"
    proc.config = {"provider_type": "gemini"}
    proc.extract.return_value = ExtractionResult(doc_id="x", model_used="stub")
    proc.summarise.return_value = SummaryResult(doc_id="x", summary="Summary.", model_used="stub")
    return proc


def _make_docs(n: int) -> list[DocumentInput]:
    return [
        DocumentInput(doc_id=f"d{i}", source_language="en", raw_text=f"Article {i}.", metadata={})
        for i in range(n)
    ]


def test_parallel_doc_processing_preserves_order_and_count(base_config):
    proc = _summarisation_processor()
    orch = PipelineOrchestrator(proc, base_config, task=PipelineTask.SUMMARISATION, model_key="stub-model")
    docs = _make_docs(5)

    results = orch.run(docs)

    assert len(results) == 5
    assert [r.document.doc_id for r in results] == [d.doc_id for d in docs]


def test_parallel_doc_processing_runs_concurrently(base_config):
    barrier = threading.Barrier(2)
    proc = _summarisation_processor()

    def slow_summarise(doc):
        barrier.wait(timeout=2)
        return SummaryResult(doc_id=doc.doc_id, summary="S", model_used="stub")

    proc.summarise.side_effect = slow_summarise
    orch = PipelineOrchestrator(proc, base_config, task=PipelineTask.SUMMARISATION, model_key="stub-model")

    results = orch.run(_make_docs(2))

    assert len(results) == 2
    assert proc.summarise.call_count == 2


def test_skip_extraction_skips_extract_on_summarisation(base_config):
    base_config["pipeline"]["skip_extraction"] = True
    proc = _summarisation_processor()
    orch = PipelineOrchestrator(proc, base_config, task=PipelineTask.SUMMARISATION, model_key="stub-model")

    orch.run(_make_docs(2))

    proc.extract.assert_not_called()
    assert proc.summarise.call_count == 2


def test_skip_extraction_ignored_for_translation(base_config):
    base_config["pipeline"]["skip_extraction"] = True
    base_config["pipeline"]["max_concurrent_documents"] = 1
    proc = MagicMock()
    proc.model_name = "stub-model"
    proc.config = {"provider_type": "gemini"}
    proc.extract.return_value = ExtractionResult(doc_id="x", model_used="stub")
    proc.translate.return_value = TranslationResult(
        doc_id="d0",
        source_language="de",
        target_language="en",
        original_text="Text",
        translated_text="Translated.",
        model_used="stub",
    )

    orch = PipelineOrchestrator(proc, base_config, task=PipelineTask.TRANSLATION, model_key="stub-model")
    orch.run([DocumentInput(doc_id="d0", source_language="de", raw_text="Text", metadata={})])

    proc.extract.assert_called_once()
