"""Unit tests for BenchmarkSampleLoader and export_benchmark_samples script."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core.models import DocumentInput
from src.pipeline.benchmark_sample_loader import BenchmarkSampleLoader


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_dir(tmp_path: Path) -> Path:
    """Write minimal well-formed sample files and return the directory."""
    translation_docs = [
        {
            "doc_id": f"europarl_de-en_{i:04d}",
            "source_language": "de",
            "raw_text": f"German text {i}",
            "source": "europarl",
            "metadata": {"reference_translation": f"English text {i}"},
        }
        for i in range(5)
    ]
    summarisation_docs = [
        {
            "doc_id": f"cnn_dm_{i:04d}",
            "source_language": "en",
            "raw_text": f"Article text {i}",
            "source": "cnn_dailymail",
            "metadata": {"reference_summary": f"Summary {i}", "cnn_id": str(i)},
        }
        for i in range(5)
    ]

    (tmp_path / "translation_de_en.json").write_text(
        json.dumps(translation_docs, ensure_ascii=False), encoding="utf-8"
    )
    (tmp_path / "summarisation_en.json").write_text(
        json.dumps(summarisation_docs, ensure_ascii=False), encoding="utf-8"
    )
    return tmp_path


# ---------------------------------------------------------------------------
# BenchmarkSampleLoader.load()
# ---------------------------------------------------------------------------

def test_load_translation_returns_document_inputs(sample_dir: Path) -> None:
    loader = BenchmarkSampleLoader(samples_dir=sample_dir)
    docs = loader.load("translation")

    assert len(docs) == 5
    assert all(isinstance(d, DocumentInput) for d in docs)


def test_load_summarisation_returns_document_inputs(sample_dir: Path) -> None:
    loader = BenchmarkSampleLoader(samples_dir=sample_dir)
    docs = loader.load("summarisation")

    assert len(docs) == 5
    assert all(isinstance(d, DocumentInput) for d in docs)


def test_load_translation_preserves_doc_id(sample_dir: Path) -> None:
    loader = BenchmarkSampleLoader(samples_dir=sample_dir)
    docs = loader.load("translation")
    assert docs[0].doc_id == "europarl_de-en_0000"


def test_load_translation_preserves_raw_text(sample_dir: Path) -> None:
    loader = BenchmarkSampleLoader(samples_dir=sample_dir)
    docs = loader.load("translation")
    assert docs[2].raw_text == "German text 2"


def test_load_translation_preserves_source_language(sample_dir: Path) -> None:
    loader = BenchmarkSampleLoader(samples_dir=sample_dir)
    docs = loader.load("translation")
    assert all(d.source_language == "de" for d in docs)


def test_load_raises_for_unknown_task(sample_dir: Path) -> None:
    loader = BenchmarkSampleLoader(samples_dir=sample_dir)
    with pytest.raises(ValueError, match="Unknown task"):
        loader.load("extraction")


def test_load_raises_when_file_missing(tmp_path: Path) -> None:
    loader = BenchmarkSampleLoader(samples_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="Benchmark sample file not found"):
        loader.load("translation")


# ---------------------------------------------------------------------------
# BenchmarkSampleLoader.ground_truth()
# ---------------------------------------------------------------------------

def test_ground_truth_translation_returns_dict(sample_dir: Path) -> None:
    loader = BenchmarkSampleLoader(samples_dir=sample_dir)
    gt = loader.ground_truth("translation")

    assert isinstance(gt, dict)
    assert len(gt) == 5


def test_ground_truth_translation_key_is_doc_id(sample_dir: Path) -> None:
    loader = BenchmarkSampleLoader(samples_dir=sample_dir)
    gt = loader.ground_truth("translation")
    assert "europarl_de-en_0000" in gt


def test_ground_truth_translation_value_is_reference(sample_dir: Path) -> None:
    loader = BenchmarkSampleLoader(samples_dir=sample_dir)
    gt = loader.ground_truth("translation")
    assert gt["europarl_de-en_0003"] == "English text 3"


def test_ground_truth_summarisation_returns_summaries(sample_dir: Path) -> None:
    loader = BenchmarkSampleLoader(samples_dir=sample_dir)
    gt = loader.ground_truth("summarisation")
    assert gt["cnn_dm_0001"] == "Summary 1"


def test_ground_truth_raises_for_unknown_task(sample_dir: Path) -> None:
    loader = BenchmarkSampleLoader(samples_dir=sample_dir)
    with pytest.raises(ValueError, match="No ground-truth key configured"):
        loader.ground_truth("full")


# ---------------------------------------------------------------------------
# BenchmarkSampleLoader.available_tasks()
# ---------------------------------------------------------------------------

def test_available_tasks_both_present(sample_dir: Path) -> None:
    loader = BenchmarkSampleLoader(samples_dir=sample_dir)
    tasks = loader.available_tasks()
    assert "translation" in tasks
    assert "summarisation" in tasks


def test_available_tasks_partial(tmp_path: Path) -> None:
    (tmp_path / "translation_de_en.json").write_text("[]", encoding="utf-8")
    loader = BenchmarkSampleLoader(samples_dir=tmp_path)
    tasks = loader.available_tasks()
    assert "translation" in tasks
    assert "summarisation" not in tasks


def test_available_tasks_empty_dir(tmp_path: Path) -> None:
    loader = BenchmarkSampleLoader(samples_dir=tmp_path)
    assert loader.available_tasks() == []


# ---------------------------------------------------------------------------
# Committed sample files (integration — run against real data/benchmark_samples/)
# ---------------------------------------------------------------------------

def test_committed_translation_samples_schema() -> None:
    """Committed translation_de_en.json must be valid and non-empty."""
    loader = BenchmarkSampleLoader()
    docs = loader.load("translation")

    assert len(docs) > 0
    for doc in docs:
        assert doc.source_language == "de"
        assert doc.raw_text
        assert "reference_translation" in doc.metadata
        assert doc.metadata["reference_translation"]


def test_committed_summarisation_samples_schema() -> None:
    """Committed summarisation_en.json must be valid and non-empty."""
    loader = BenchmarkSampleLoader()
    docs = loader.load("summarisation")

    assert len(docs) > 0
    for doc in docs:
        assert doc.source_language == "en"
        assert doc.raw_text
        assert "reference_summary" in doc.metadata
        assert doc.metadata["reference_summary"]


def test_committed_translation_ground_truth_completeness() -> None:
    loader = BenchmarkSampleLoader()
    docs = loader.load("translation")
    gt = loader.ground_truth("translation")
    assert set(gt.keys()) == {d.doc_id for d in docs}


def test_committed_summarisation_ground_truth_completeness() -> None:
    loader = BenchmarkSampleLoader()
    docs = loader.load("summarisation")
    gt = loader.ground_truth("summarisation")
    assert set(gt.keys()) == {d.doc_id for d in docs}
