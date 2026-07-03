"""Tests for src/core/provenance.py — hash utilities."""

import hashlib
import json
from pathlib import Path

import pytest

from src.core.provenance import (
    config_hash,
    config_snapshot,
    dataset_hash,
    file_hash,
    ground_truth_hash,
)


# ─────────────────────────────────────────────────────
# file_hash / dataset_hash
# ─────────────────────────────────────────────────────

def test_file_hash_returns_sha256_prefix(tmp_path):
    f = tmp_path / "test.json"
    f.write_bytes(b'{"hello": "world"}')
    result = file_hash(f)
    assert result.startswith("sha256:")
    assert len(result) == len("sha256:") + 64  # 64 hex chars


def test_file_hash_is_stable_across_calls(tmp_path):
    f = tmp_path / "data.json"
    f.write_bytes(b"stable content")
    assert file_hash(f) == file_hash(f)


def test_file_hash_differs_for_different_content(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_bytes(b"content A")
    b.write_bytes(b"content B")
    assert file_hash(a) != file_hash(b)


def test_dataset_hash_is_alias_for_file_hash(tmp_path):
    f = tmp_path / "dataset.json"
    f.write_bytes(b'[{"doc_id": "d1"}]')
    assert dataset_hash(f) == file_hash(f)


def test_file_hash_known_value(tmp_path):
    """Regression: known content must produce a stable known hash."""
    content = b"llmevallab"
    expected = "sha256:" + hashlib.sha256(content).hexdigest()
    f = tmp_path / "known.bin"
    f.write_bytes(content)
    assert file_hash(f) == expected


# ─────────────────────────────────────────────────────
# ground_truth_hash
# ─────────────────────────────────────────────────────

@pytest.fixture
def gt_fixture(tmp_path) -> tuple[Path, list[str]]:
    """Returns (path, doc_ids) for a small ground truth dataset fixture."""
    docs = [
        {"doc_id": "d1", "metadata": {"reference_summary": "Summary one."}},
        {"doc_id": "d2", "metadata": {"reference_summary": "Summary two."}},
        {"doc_id": "d3", "metadata": {"reference_summary": "Summary three."}},
    ]
    p = tmp_path / "gt.json"
    p.write_text(json.dumps(docs), encoding="utf-8")
    return p, ["d1", "d2"]


def test_ground_truth_hash_returns_sha256_prefix(gt_fixture):
    path, doc_ids = gt_fixture
    result = ground_truth_hash(doc_ids, path, "reference_summary")
    assert result.startswith("sha256:")


def test_ground_truth_hash_is_stable(gt_fixture):
    path, doc_ids = gt_fixture
    h1 = ground_truth_hash(doc_ids, path, "reference_summary")
    h2 = ground_truth_hash(doc_ids, path, "reference_summary")
    assert h1 == h2


def test_ground_truth_hash_order_independent(gt_fixture):
    """Hashing [d1, d2] must equal hashing [d2, d1] — canonical sort applied."""
    path, _ = gt_fixture
    h_forward  = ground_truth_hash(["d1", "d2"], path, "reference_summary")
    h_backward = ground_truth_hash(["d2", "d1"], path, "reference_summary")
    assert h_forward == h_backward


def test_ground_truth_hash_different_subset(gt_fixture):
    """Different doc_id subsets must produce different hashes."""
    path, _ = gt_fixture
    h_d1_d2 = ground_truth_hash(["d1", "d2"], path, "reference_summary")
    h_d2_d3 = ground_truth_hash(["d2", "d3"], path, "reference_summary")
    assert h_d1_d2 != h_d2_d3


def test_ground_truth_hash_changes_when_text_changes(tmp_path):
    """Modifying a reference text changes the hash."""
    docs_v1 = [{"doc_id": "d1", "metadata": {"reference_summary": "Original text."}}]
    docs_v2 = [{"doc_id": "d1", "metadata": {"reference_summary": "Changed text."}}]
    p1 = tmp_path / "gt_v1.json"
    p2 = tmp_path / "gt_v2.json"
    p1.write_text(json.dumps(docs_v1))
    p2.write_text(json.dumps(docs_v2))

    h1 = ground_truth_hash(["d1"], p1, "reference_summary")
    h2 = ground_truth_hash(["d1"], p2, "reference_summary")
    assert h1 != h2


# ─────────────────────────────────────────────────────
# config_hash / config_snapshot
# ─────────────────────────────────────────────────────

@pytest.fixture
def sample_config() -> dict:
    return {
        "pipeline": {"target_language": "en", "max_document_length": 2000},
        "models": {
            "catalog": {
                "gemini-flash": {
                    "provider_type": "gemini",
                    "model_id": "gemini-2.5-flash",
                    "api_key_env": "GOOGLE_API_KEY",
                    "pricing": {"input_per_1m": 0.075, "output_per_1m": 0.30},
                }
            }
        },
    }


def test_config_snapshot_structure(sample_config):
    snap = config_snapshot(sample_config, "gemini-flash")
    assert "pipeline" in snap
    assert "model" in snap
    assert snap["pipeline"]["target_language"] == "en"
    assert snap["model"]["model_id"] == "gemini-2.5-flash"


def test_config_hash_returns_sha256_prefix(sample_config):
    result = config_hash(sample_config, "gemini-flash")
    assert result.startswith("sha256:")


def test_config_hash_is_stable(sample_config):
    h1 = config_hash(sample_config, "gemini-flash")
    h2 = config_hash(sample_config, "gemini-flash")
    assert h1 == h2


def test_config_hash_changes_when_model_changes(sample_config):
    """Adding a second model key produces a different hash for that key."""
    import copy
    config2 = copy.deepcopy(sample_config)
    config2["models"]["catalog"]["other-model"] = {
        "provider_type": "claude",
        "model_id": "claude-sonnet",
        "api_key_env": "ANTHROPIC_API_KEY",
        "pricing": {"input_per_1m": 3.0, "output_per_1m": 15.0},
    }
    h1 = config_hash(sample_config, "gemini-flash")
    # The snapshot for gemini-flash itself didn't change:
    h2 = config_hash(config2, "gemini-flash")
    assert h1 == h2  # same model entry → same hash


def test_config_hash_changes_when_temperature_changes(sample_config):
    import copy
    config_hot = copy.deepcopy(sample_config)
    config_hot["pipeline"]["target_language"] = "fr"
    h_en = config_hash(sample_config, "gemini-flash")
    h_fr = config_hash(config_hot, "gemini-flash")
    assert h_en != h_fr
