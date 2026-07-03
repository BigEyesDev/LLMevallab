"""Tests for Priority 3: task-specific truncation limits and metadata.

Covers:
- config.yaml has per-task limits readable by the orchestrator
- TruncationInfo is populated correctly (truncated vs not)
- Manifest config_snapshot captures the effective truncation limit
- All three task values use their own limits
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.core.models import (
    DocumentInput,
    ExtractionResult,
    PipelineResult,
    RunManifest,
    SummaryResult,
    TranslationResult,
    TruncationInfo,
    DEFAULT_TASK_TRUNCATION_LIMITS,
)
from src.pipeline.orchestrator import PipelineOrchestrator, PipelineTask


# ─────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────

@pytest.fixture
def base_config(tmp_path):
    return {
        "pipeline": {
            "target_language": "en",
            "max_document_length": 2000,
            "max_document_length_per_task": {
                "translation": 2000,
                "summarisation": 8000,
                "full": 4000,
            },
        },
        "paths": {"outputs": str(tmp_path)},
        "models": {"catalog": {"stub-model": {"model_id": "stub"}}},
    }


@pytest.fixture
def stub_summarisation_processor():
    proc = MagicMock()
    proc.model_name = "stub-model"

    def extract(doc):
        return ExtractionResult(doc_id=doc.doc_id, model_used="stub-model")

    def summarise(doc):
        return SummaryResult(doc_id=doc.doc_id, summary="A summary.", model_used="stub-model")

    proc.extract.side_effect = extract
    proc.summarise.side_effect = summarise
    return proc


@pytest.fixture
def stub_translation_processor():
    proc = MagicMock()
    proc.model_name = "stub-model"

    def extract(doc):
        return ExtractionResult(doc_id=doc.doc_id, model_used="stub-model")

    def translate(doc, target_language="en"):
        return TranslationResult(
            doc_id=doc.doc_id,
            source_language=doc.source_language,
            original_text=doc.raw_text,
            translated_text="Translated.",
            model_used="stub-model",
        )

    proc.extract.side_effect = extract
    proc.translate.side_effect = translate
    return proc


# ─────────────────────────────────────────────────────
# TruncationInfo model
# ─────────────────────────────────────────────────────

class TestTruncationInfo:
    def test_not_truncated(self):
        info = TruncationInfo(
            chars_original=100,
            chars_sent=100,
            was_truncated=False,
            limit_applied=2000,
        )
        assert not info.was_truncated
        assert info.chars_original == info.chars_sent

    def test_truncated(self):
        info = TruncationInfo(
            chars_original=5000,
            chars_sent=2000,
            was_truncated=True,
            limit_applied=2000,
        )
        assert info.was_truncated
        assert info.chars_sent < info.chars_original

    def test_roundtrip(self):
        info = TruncationInfo(
            chars_original=9000,
            chars_sent=8000,
            was_truncated=True,
            limit_applied=8000,
        )
        data = info.model_dump()
        restored = TruncationInfo.model_validate(data)
        assert restored.chars_original == 9000
        assert restored.limit_applied == 8000


# ─────────────────────────────────────────────────────
# DEFAULT_TASK_TRUNCATION_LIMITS
# ─────────────────────────────────────────────────────

class TestDefaultTruncationLimits:
    def test_all_tasks_have_defaults(self):
        for task in ("translation", "summarisation", "full"):
            assert task in DEFAULT_TASK_TRUNCATION_LIMITS
            assert DEFAULT_TASK_TRUNCATION_LIMITS[task] > 0

    def test_summarisation_limit_larger_than_translation(self):
        assert DEFAULT_TASK_TRUNCATION_LIMITS["summarisation"] > DEFAULT_TASK_TRUNCATION_LIMITS["translation"]


# ─────────────────────────────────────────────────────
# Orchestrator: per-task limit resolution
# ─────────────────────────────────────────────────────

class TestOrchestratorTruncationLimit:
    def test_summarisation_uses_per_task_limit(self, base_config, stub_summarisation_processor):
        orch = PipelineOrchestrator(
            processor=stub_summarisation_processor,
            config=base_config,
            task=PipelineTask.SUMMARISATION,
        )
        assert orch._get_task_truncation_limit() == 8000

    def test_translation_uses_per_task_limit(self, base_config, stub_translation_processor):
        orch = PipelineOrchestrator(
            processor=stub_translation_processor,
            config=base_config,
            task=PipelineTask.TRANSLATION,
        )
        assert orch._get_task_truncation_limit() == 2000

    def test_full_uses_per_task_limit(self, base_config, stub_summarisation_processor):
        orch = PipelineOrchestrator(
            processor=stub_summarisation_processor,
            config=base_config,
            task=PipelineTask.FULL,
        )
        assert orch._get_task_truncation_limit() == 4000

    def test_falls_back_to_global_limit_when_per_task_missing(self, tmp_path, stub_summarisation_processor):
        config = {
            "pipeline": {"target_language": "en", "max_document_length": 1500},
            "paths": {"outputs": str(tmp_path)},
            "models": {"catalog": {}},
        }
        orch = PipelineOrchestrator(
            processor=stub_summarisation_processor,
            config=config,
            task=PipelineTask.SUMMARISATION,
        )
        assert orch._get_task_truncation_limit() == 1500

    def test_falls_back_to_hardcoded_defaults_when_no_config_limit(self, tmp_path, stub_summarisation_processor):
        config = {
            "pipeline": {"target_language": "en"},
            "paths": {"outputs": str(tmp_path)},
            "models": {"catalog": {}},
        }
        orch = PipelineOrchestrator(
            processor=stub_summarisation_processor,
            config=config,
            task=PipelineTask.SUMMARISATION,
        )
        assert orch._get_task_truncation_limit() == DEFAULT_TASK_TRUNCATION_LIMITS["summarisation"]


# ─────────────────────────────────────────────────────
# Orchestrator: _truncate_document
# ─────────────────────────────────────────────────────

class TestOrchestratorTruncateDocument:
    def _make_orch(self, base_config, processor, task=PipelineTask.SUMMARISATION):
        return PipelineOrchestrator(
            processor=processor,
            config=base_config,
            task=task,
        )

    def test_short_doc_not_truncated(self, base_config, stub_summarisation_processor):
        orch = self._make_orch(base_config, stub_summarisation_processor)
        doc = DocumentInput(doc_id="d1", source_language="en", raw_text="Short text.")
        result_doc, info = orch._truncate_document(doc, limit=8000)
        assert not info.was_truncated
        assert result_doc.raw_text == "Short text."
        assert info.chars_sent == info.chars_original

    def test_long_doc_is_truncated(self, base_config, stub_summarisation_processor):
        orch = self._make_orch(base_config, stub_summarisation_processor)
        long_text = "x" * 10000
        doc = DocumentInput(doc_id="d1", source_language="en", raw_text=long_text)
        result_doc, info = orch._truncate_document(doc, limit=8000)
        assert info.was_truncated
        assert len(result_doc.raw_text) == 8000
        assert info.chars_sent == 8000
        assert info.chars_original == 10000

    def test_truncation_does_not_mutate_original_doc(self, base_config, stub_summarisation_processor):
        orch = self._make_orch(base_config, stub_summarisation_processor)
        long_text = "y" * 5000
        doc = DocumentInput(doc_id="d1", source_language="en", raw_text=long_text)
        _, _ = orch._truncate_document(doc, limit=2000)
        assert len(doc.raw_text) == 5000, "Original document must not be mutated"


# ─────────────────────────────────────────────────────
# PipelineResult carries TruncationInfo after run()
# ─────────────────────────────────────────────────────

class TestPipelineResultTruncation:
    def test_truncation_info_present_on_result(self, base_config, stub_summarisation_processor):
        docs = [DocumentInput(doc_id="d1", source_language="en", raw_text="Hello world.")]
        orch = PipelineOrchestrator(
            processor=stub_summarisation_processor,
            config=base_config,
            task=PipelineTask.SUMMARISATION,
        )
        results = orch.run(docs)
        assert results[0].truncation is not None

    def test_short_doc_not_truncated_in_result(self, base_config, stub_summarisation_processor):
        docs = [DocumentInput(doc_id="d1", source_language="en", raw_text="Short.")]
        orch = PipelineOrchestrator(
            processor=stub_summarisation_processor,
            config=base_config,
            task=PipelineTask.SUMMARISATION,
        )
        results = orch.run(docs)
        assert not results[0].truncation.was_truncated

    def test_long_doc_truncated_in_result(self, base_config, stub_summarisation_processor):
        long_text = "z" * 12000
        docs = [DocumentInput(doc_id="d1", source_language="en", raw_text=long_text)]
        orch = PipelineOrchestrator(
            processor=stub_summarisation_processor,
            config=base_config,
            task=PipelineTask.SUMMARISATION,
        )
        results = orch.run(docs)
        trunc = results[0].truncation
        assert trunc.was_truncated
        assert trunc.chars_original == 12000
        assert trunc.chars_sent == 8000

    def test_original_document_text_preserved_in_result(self, base_config, stub_summarisation_processor):
        long_text = "a" * 10000
        docs = [DocumentInput(doc_id="d1", source_language="en", raw_text=long_text)]
        orch = PipelineOrchestrator(
            processor=stub_summarisation_processor,
            config=base_config,
            task=PipelineTask.SUMMARISATION,
        )
        results = orch.run(docs)
        assert len(results[0].document.raw_text) == 10000, "Original full text must be stored in PipelineResult"


# ─────────────────────────────────────────────────────
# Manifest config_snapshot includes truncation limit
# ─────────────────────────────────────────────────────

class TestManifestTruncationSnapshot:
    def test_manifest_snapshot_includes_truncation_limit(self, tmp_path, stub_summarisation_processor):
        config = {
            "pipeline": {
                "target_language": "en",
                "max_document_length": 2000,
                "max_document_length_per_task": {
                    "translation": 2000,
                    "summarisation": 8000,
                    "full": 4000,
                },
            },
            "paths": {"outputs": str(tmp_path)},
            "models": {
                "catalog": {"test-model": {"model_id": "test-model"}},
                "default": "test-model",
            },
        }

        docs = [DocumentInput(doc_id="d1", source_language="en", raw_text="Article text.")]
        orch = PipelineOrchestrator(
            processor=stub_summarisation_processor,
            config=config,
            task=PipelineTask.SUMMARISATION,
            model_key="test-model",
        )
        orch.run(docs)

        manifest_file = next(tmp_path.glob("results_summarisation_*.manifest.json"))
        manifest = RunManifest.model_validate(json.loads(manifest_file.read_text()))

        assert "truncation_limit_applied" in manifest.config_snapshot
        assert manifest.config_snapshot["truncation_limit_applied"] == 8000
        assert manifest.config_snapshot["task"] == "summarisation"
