"""Tests for RunManifest DTO, orchestrator manifest writing, and evaluator hash verification."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.models import (
    DocumentInput,
    EvaluationReport,
    EvaluationScore,
    PipelineResult,
    RunManifest,
    SummaryResult,
)
from src.core.provenance import ground_truth_hash
from src.evaluations.evaluator import (
    Evaluator,
    verify_manifest_ground_truth,
    audit_doc_ids,
)
from src.pipeline.orchestrator import PipelineOrchestrator, PipelineTask


# ─────────────────────────────────────────────────────
# RunManifest DTO
# ─────────────────────────────────────────────────────

def test_run_manifest_roundtrip():
    """RunManifest must survive JSON → dict → model round-trips."""
    manifest = RunManifest(
        run_id="20260703_094208_a1b2c3",
        app_version="0.2.0",
        task="summarisation",
        model_key="gemini-2.5-flash",
        model_id="gemini-2.5-flash",
        dataset_path="data/processed/cnn_dailymail/cnn_dailymail_20docs.json",
        dataset_hash="sha256:abc123",
        doc_ids=["cnn_0000", "cnn_0001"],
        sample_size=2,
        sample_indices=[0, 1],
        ground_truth_path="data/processed/cnn_dailymail/cnn_dailymail_20docs.json",
        ground_truth_hash="sha256:def456",
        config_hash="sha256:789abc",
        config_snapshot={"pipeline": {"target_language": "en"}},
        results_path="outputs/results/results_summarisation_gemini_...json",
    )
    data = manifest.model_dump()
    restored = RunManifest.model_validate(data)

    assert restored.run_id == manifest.run_id
    assert restored.task == manifest.task
    assert restored.doc_ids == manifest.doc_ids
    assert restored.ground_truth_hash == manifest.ground_truth_hash
    assert restored.config_snapshot == manifest.config_snapshot


def test_run_manifest_json_roundtrip(tmp_path):
    """RunManifest must survive serialisation through a real JSON file."""
    manifest = RunManifest(
        run_id="20260703_123456_deadbe",
        app_version="0.2.0",
        task="translation",
        model_key="gemini-2.5-flash",
        model_id="gemini-2.5-flash",
        dataset_path="/data/europarl.json",
        dataset_hash="sha256:aaa",
        doc_ids=["ep_0001"],
        sample_size=1,
        sample_indices=[0],
        ground_truth_path="/data/europarl.json",
        ground_truth_hash="sha256:bbb",
        config_hash="sha256:ccc",
        config_snapshot={},
        results_path="/out/results.json",
    )
    p = tmp_path / "run.manifest.json"
    p.write_text(json.dumps(manifest.model_dump(), default=str))

    loaded = RunManifest.model_validate(json.loads(p.read_text()))
    assert loaded.run_id == manifest.run_id
    assert loaded.sample_size == 1


def test_evaluation_report_has_run_id_field():
    """EvaluationReport must carry optional run_id and manifest_path."""
    report = EvaluationReport(
        model_used="test-model",
        run_id="20260703_094208_a1b2c3",
        manifest_path="/out/run.manifest.json",
    )
    assert report.run_id == "20260703_094208_a1b2c3"
    assert report.manifest_path == "/out/run.manifest.json"


def test_evaluation_report_run_id_optional():
    """EvaluationReport must be constructable without run_id (backward-compat)."""
    report = EvaluationReport(model_used="test-model")
    assert report.run_id is None
    assert report.manifest_path is None


# ─────────────────────────────────────────────────────
# Orchestrator writes manifest
# ─────────────────────────────────────────────────────

@pytest.fixture
def minimal_docs():
    return [
        DocumentInput(doc_id="d1", source_language="en", raw_text="Article one."),
        DocumentInput(doc_id="d2", source_language="en", raw_text="Article two."),
    ]


@pytest.fixture
def stub_processor():
    """A minimal processor mock that returns SummaryResult for each doc."""
    from src.core.models import ExtractionResult

    proc = MagicMock()
    proc.model_name = "stub-model"

    def extract(doc):
        return ExtractionResult(doc_id=doc.doc_id, model_used="stub-model")

    def summarise(doc):
        return SummaryResult(doc_id=doc.doc_id, summary="A summary.", model_used="stub-model")

    proc.extract.side_effect = extract
    proc.summarise.side_effect = summarise
    return proc


def test_manifest_written_alongside_results(tmp_path, minimal_docs, stub_processor):
    """Orchestrator must write a *.manifest.json next to every results file."""
    config = {
        "pipeline": {"target_language": "en", "max_document_length": 2000},
        "paths": {"outputs": str(tmp_path)},
        "models": {"catalog": {"stub-model": {"model_id": "stub"}}},
    }

    orch = PipelineOrchestrator(
        processor=stub_processor,
        config=config,
        task=PipelineTask.SUMMARISATION,
    )
    orch.run(minimal_docs)

    manifest_files = list(tmp_path.glob("results_summarisation_*.manifest.json"))
    results_files = [
        f for f in tmp_path.glob("results_summarisation_*.json")
        if not f.name.endswith(".manifest.json")
    ]

    assert len(results_files) == 1, "Expected exactly one results JSON"
    assert len(manifest_files) == 1, "Expected exactly one manifest JSON"


def test_manifest_fields_populated(tmp_path, minimal_docs, stub_processor):
    """Manifest must contain correct doc_ids, sample_size, task, and app_version."""
    config = {
        "pipeline": {"target_language": "en", "max_document_length": 2000},
        "paths": {"outputs": str(tmp_path)},
        "models": {"catalog": {"stub-model": {"model_id": "stub"}}},
    }

    orch = PipelineOrchestrator(
        processor=stub_processor,
        config=config,
        task=PipelineTask.SUMMARISATION,
    )
    orch.run(minimal_docs)

    manifest_file = next(tmp_path.glob("results_summarisation_*.manifest.json"))
    manifest = RunManifest.model_validate(json.loads(manifest_file.read_text()))

    assert set(manifest.doc_ids) == {"d1", "d2"}
    assert manifest.sample_size == 2
    assert manifest.task == "summarisation"
    assert manifest.app_version  # non-empty


def test_manifest_with_dataset_path_hashes(tmp_path, minimal_docs, stub_processor):
    """When dataset_path is provided with real data, dataset_hash must be non-empty."""
    # Write a minimal dataset file
    dataset = [
        {"doc_id": "d1", "raw_text": "Article one.", "metadata": {}},
        {"doc_id": "d2", "raw_text": "Article two.", "metadata": {}},
    ]
    dataset_file = tmp_path / "docs.json"
    dataset_file.write_text(json.dumps(dataset))

    config = {
        "pipeline": {"target_language": "en", "max_document_length": 2000},
        "paths": {"outputs": str(tmp_path)},
        "models": {"catalog": {"stub-model": {"model_id": "stub"}}},
    }

    orch = PipelineOrchestrator(
        processor=stub_processor,
        config=config,
        task=PipelineTask.SUMMARISATION,
        dataset_path=str(dataset_file),
    )
    orch.run(minimal_docs)

    manifest_file = next(tmp_path.glob("results_summarisation_*.manifest.json"))
    manifest = RunManifest.model_validate(json.loads(manifest_file.read_text()))

    assert manifest.dataset_hash.startswith("sha256:")


# ─────────────────────────────────────────────────────
# Evaluator — manifest verification
# ─────────────────────────────────────────────────────

@pytest.fixture
def gt_dataset(tmp_path) -> Path:
    """Minimal ground truth dataset with reference_summary."""
    docs = [
        {"doc_id": "d1", "metadata": {"reference_summary": "Reference one."}},
        {"doc_id": "d2", "metadata": {"reference_summary": "Reference two."}},
    ]
    p = tmp_path / "gt.json"
    p.write_text(json.dumps(docs))
    return p


def _make_manifest(tmp_path, gt_path: Path, doc_ids: list[str], hash_override=None) -> Path:
    """Helper: write a manifest JSON for testing."""
    gt_hash = ground_truth_hash(doc_ids, gt_path, "reference_summary") if hash_override is None else hash_override
    manifest = RunManifest(
        run_id="20260703_TEST_abc",
        app_version="0.2.0",
        task="summarisation",
        model_key="test-model",
        model_id="test-model",
        dataset_path=str(gt_path),
        dataset_hash="sha256:aaa",
        doc_ids=doc_ids,
        sample_size=len(doc_ids),
        sample_indices=list(range(len(doc_ids))),
        ground_truth_path=str(gt_path),
        ground_truth_hash=gt_hash,
        config_hash="sha256:ccc",
        config_snapshot={},
        results_path=str(tmp_path / "results.json"),
    )
    p = tmp_path / "run.manifest.json"
    p.write_text(json.dumps(manifest.model_dump(), default=str))
    return p


def test_verify_manifest_ground_truth_passes_on_correct_hash(tmp_path, gt_dataset):
    """verify_manifest_ground_truth must not raise when hash matches."""
    manifest_path = _make_manifest(tmp_path, gt_dataset, ["d1", "d2"])
    manifest = RunManifest.model_validate(json.loads(manifest_path.read_text()))
    verify_manifest_ground_truth(manifest)  # must not raise


def test_verify_manifest_ground_truth_raises_on_wrong_hash(tmp_path, gt_dataset):
    """verify_manifest_ground_truth must raise RuntimeError when hash does not match."""
    manifest_path = _make_manifest(tmp_path, gt_dataset, ["d1", "d2"], hash_override="sha256:WRONG")
    manifest = RunManifest.model_validate(json.loads(manifest_path.read_text()))
    with pytest.raises(RuntimeError, match="Ground truth hash MISMATCH"):
        verify_manifest_ground_truth(manifest)


def test_verify_manifest_raises_on_empty_hash(tmp_path, gt_dataset):
    """verify_manifest_ground_truth must raise ValueError when hash is empty string."""
    manifest_path = _make_manifest(tmp_path, gt_dataset, ["d1", "d2"], hash_override="")
    manifest = RunManifest.model_validate(json.loads(manifest_path.read_text()))
    with pytest.raises(ValueError, match="no ground_truth_hash recorded"):
        verify_manifest_ground_truth(manifest)


def test_audit_doc_ids_warns_on_extra(caplog, tmp_path, gt_dataset):
    """audit_doc_ids must log a warning for result doc_ids not in manifest."""
    manifest_path = _make_manifest(tmp_path, gt_dataset, ["d1"])
    manifest = RunManifest.model_validate(json.loads(manifest_path.read_text()))

    pipeline_results = [
        {"document": {"doc_id": "d1"}, "summary": {}},
        {"document": {"doc_id": "d_UNKNOWN"}, "summary": {}},
    ]
    import logging
    with caplog.at_level(logging.WARNING, logger="src.evaluations.evaluator"):
        audit_doc_ids(pipeline_results, manifest)

    assert any("d_UNKNOWN" in r.message for r in caplog.records)


# ─────────────────────────────────────────────────────
# Evaluator.run_on_manifest — integration (mocked metrics)
# ─────────────────────────────────────────────────────

def test_evaluator_run_on_manifest_produces_report(tmp_path, gt_dataset):
    """run_on_manifest must return an EvaluationReport with the manifest's run_id."""
    # Write a minimal results JSON
    results = [
        {
            "document": {"doc_id": "d1", "source_language": "en", "raw_text": "", "source": "test", "metadata": {}},
            "summary": {"doc_id": "d1", "summary": "Output one.", "key_points": [], "action_items": [],
                        "model_used": "test-model", "processing_time_ms": 0.0, "token_usage": None, "cost_usd": 0.0},
            "extraction": None, "translation": None, "total_processing_time_ms": 0.0,
            "run_timestamp": "2026-07-03T09:42:08+00:00",
        },
        {
            "document": {"doc_id": "d2", "source_language": "en", "raw_text": "", "source": "test", "metadata": {}},
            "summary": {"doc_id": "d2", "summary": "Output two.", "key_points": [], "action_items": [],
                        "model_used": "test-model", "processing_time_ms": 0.0, "token_usage": None, "cost_usd": 0.0},
            "extraction": None, "translation": None, "total_processing_time_ms": 0.0,
            "run_timestamp": "2026-07-03T09:42:08+00:00",
        },
    ]
    results_file = tmp_path / "results.json"
    results_file.write_text(json.dumps(results))

    manifest_path = _make_manifest(tmp_path, gt_dataset, ["d1", "d2"])
    # Update the manifest to point at the results file we just wrote
    manifest_data = json.loads(manifest_path.read_text())
    manifest_data["results_path"] = str(results_file)
    manifest_path.write_text(json.dumps(manifest_data))

    config = {
        "paths": {"reports": str(tmp_path / "reports")},
        "evaluation": {"bertscore_model": "microsoft/deberta-xlarge-mnli"},
    }
    evaluator = Evaluator(config)

    with patch("src.evaluations.evaluator.MetricsRunner") as MockRunner, \
         patch("src.evaluations.evaluator.save_report") as mock_save:
        mock_runner_inst = MagicMock()
        MockRunner.return_value = mock_runner_inst
        mock_runner_inst.run_all.return_value = {
            "rouge": [
                EvaluationScore(doc_id="d1", metric_name="rouge", score=0.8),
                EvaluationScore(doc_id="d2", metric_name="rouge", score=0.7),
            ]
        }

        report = evaluator.run_on_manifest(str(manifest_path))

    assert report.run_id == "20260703_TEST_abc"
    assert report.manifest_path == str(manifest_path)
    assert report.model_used == "test-model"
    mock_save.assert_called_once()
