"""Tests for on-disk past run session loading."""

import json
from pathlib import Path

from src.core.models import RunManifest
from src.pipeline.run_history import load_past_run_sessions


def _write_manifest(path: Path, **overrides) -> RunManifest:
    base = {
        "run_id": "20260708_120000_abc123",
        "app_version": "0.3.1",
        "task": "summarisation",
        "model_key": "gemini-2.5-flash",
        "model_id": "gemini-2.5-flash",
        "dataset_path": "",
        "dataset_hash": "",
        "doc_ids": ["cnn_dm_0000", "cnn_dm_0001"],
        "sample_size": 2,
        "sample_indices": [0, 1],
        "ground_truth_path": "",
        "ground_truth_hash": "",
        "config_hash": "",
        "config_snapshot": {},
        "results_path": str(path.with_suffix(".json")),
        "created_at": "2026-07-08T12:00:00+00:00",
    }
    base.update(overrides)
    manifest = RunManifest.model_validate(base)
    path.write_text(json.dumps(manifest.model_dump(), indent=2), encoding="utf-8")
    return manifest


def test_load_past_run_sessions_groups_multi_model_run(tmp_path: Path):
    out = tmp_path / "outputs"
    out.mkdir()
    _write_manifest(
        out / "run_a.manifest.json",
        model_key="gemini-2.5-flash",
        created_at="2026-07-08T12:00:00+00:00",
        results_path=str(out / "results_summarisation_gemini_2.5_flash_20260708_120000.json"),
    )
    _write_manifest(
        out / "run_b.manifest.json",
        model_key="gpt-4o-mini",
        created_at="2026-07-08T12:05:00+00:00",
        results_path=str(out / "results_summarisation_gpt_4o_mini_20260708_120500.json"),
    )

    sessions = load_past_run_sessions(out, task="summarisation", limit=10)

    assert len(sessions) == 1
    assert sessions[0].n_docs == 2
    assert sessions[0].doc_ids == ["cnn_dm_0000", "cnn_dm_0001"]
    assert set(sessions[0].models) == {"gemini-2.5-flash", "gpt-4o-mini"}


def test_load_past_run_sessions_separates_different_doc_sets(tmp_path: Path):
    out = tmp_path / "outputs"
    out.mkdir()
    _write_manifest(
        out / "run_a.manifest.json",
        doc_ids=["cnn_dm_0000"],
        created_at="2026-07-08T12:00:00+00:00",
    )
    _write_manifest(
        out / "run_b.manifest.json",
        doc_ids=["cnn_dm_0005", "cnn_dm_0006"],
        created_at="2026-07-08T13:00:00+00:00",
    )

    sessions = load_past_run_sessions(out, task="summarisation", limit=10)

    assert len(sessions) == 2
    assert sessions[0].doc_ids == ["cnn_dm_0005", "cnn_dm_0006"]
    assert sessions[1].doc_ids == ["cnn_dm_0000"]


def test_load_past_run_sessions_infers_model_from_results_filename(tmp_path: Path):
    out = tmp_path / "outputs"
    out.mkdir()
    _write_manifest(
        out / "run_a.manifest.json",
        model_key="",
        results_path=str(out / "results_summarisation_deepseek_v4_flash_20260708_120000.json"),
    )

    sessions = load_past_run_sessions(out, limit=10)

    assert sessions[0].models == ["deepseek-v4-flash"]
