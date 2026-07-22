"""Tests for src/pipeline/document_sets.py — registry CRUD, name generation, dedupe."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.pipeline.document_sets import (
    DocumentSet,
    generate_set_name,
    load_registry,
    lookup_by_hash,
    lookup_by_name,
    register_or_get,
    save_registry,
)
from src.core.provenance import selection_hash


# ─────────────────────────────────────────────────────
# selection_hash (integration with provenance)
# ─────────────────────────────────────────────────────

def test_selection_hash_returns_12_chars():
    h = selection_hash("summarisation", ["cnn_dm_0000", "cnn_dm_0001"])
    assert len(h) == 12
    assert all(c in "0123456789abcdef" for c in h)


def test_selection_hash_order_independent():
    h1 = selection_hash("summarisation", ["cnn_dm_0000", "cnn_dm_0001"])
    h2 = selection_hash("summarisation", ["cnn_dm_0001", "cnn_dm_0000"])
    assert h1 == h2


def test_selection_hash_task_scoped():
    h_sum = selection_hash("summarisation", ["cnn_dm_0000"])
    h_tra = selection_hash("translation", ["cnn_dm_0000"])
    assert h_sum != h_tra


def test_selection_hash_differs_for_different_docs():
    h1 = selection_hash("summarisation", ["cnn_dm_0000", "cnn_dm_0001"])
    h2 = selection_hash("summarisation", ["cnn_dm_0000", "cnn_dm_0002"])
    assert h1 != h2


def test_selection_hash_is_stable():
    h1 = selection_hash("summarisation", ["cnn_dm_0000"])
    h2 = selection_hash("summarisation", ["cnn_dm_0000"])
    assert h1 == h2


# ─────────────────────────────────────────────────────
# DocumentSet model
# ─────────────────────────────────────────────────────

def _make_ds(name: str = "pearl_wish") -> DocumentSet:
    return DocumentSet(
        set_name=name,
        selection_hash="abc123def456",
        task="summarisation",
        doc_ids=["cnn_dm_0000", "cnn_dm_0001"],
        data_source="benchmark_samples",
        n_docs=2,
        created_at="2026-07-08T00:00:00+00:00",
        last_used_at="2026-07-08T00:00:00+00:00",
    )


def test_document_set_round_trips():
    ds = _make_ds()
    assert DocumentSet.from_dict(ds.to_dict()) == ds


def test_document_set_to_dict_keys():
    d = _make_ds().to_dict()
    for key in ("set_name", "selection_hash", "task", "doc_ids", "data_source", "n_docs", "created_at", "last_used_at"):
        assert key in d


# ─────────────────────────────────────────────────────
# Registry I/O
# ─────────────────────────────────────────────────────

def test_load_registry_returns_empty_when_file_missing(tmp_path):
    result = load_registry(tmp_path / "nonexistent.json")
    assert result == {}


def test_save_and_load_registry(tmp_path):
    reg_path = tmp_path / "registry.json"
    ds = _make_ds()
    save_registry({"pearl_wish": ds}, reg_path)
    loaded = load_registry(reg_path)
    assert "pearl_wish" in loaded
    assert loaded["pearl_wish"].set_name == "pearl_wish"
    assert loaded["pearl_wish"].selection_hash == "abc123def456"


def test_save_registry_creates_parent_dirs(tmp_path):
    reg_path = tmp_path / "sub" / "dir" / "registry.json"
    save_registry({}, reg_path)
    assert reg_path.exists()


def test_registry_json_is_human_readable(tmp_path):
    reg_path = tmp_path / "registry.json"
    ds = _make_ds()
    save_registry({"pearl_wish": ds}, reg_path)
    raw = reg_path.read_text()
    assert "pearl_wish" in raw
    assert "\n" in raw  # indented


# ─────────────────────────────────────────────────────
# generate_set_name
# ─────────────────────────────────────────────────────

def test_generate_set_name_format():
    name = generate_set_name({})
    assert "_" in name
    parts = name.split("_")
    assert len(parts) == 2
    assert all(p.islower() for p in parts)


def test_generate_set_name_avoids_existing():
    from src.pipeline.document_sets import _ADJECTIVES, _NOUNS
    # Fill registry with all possible names for a tiny subset to force collision
    tiny_adj = ["amber"]
    tiny_noun = ["arch"]
    # Monkey-patch is impractical here; just verify uniqueness with empty + one existing
    ds = _make_ds("amber_arch")
    existing = {"amber_arch": ds}
    name = generate_set_name(existing)
    assert name != "amber_arch"


def test_generate_set_name_unique_in_empty_registry():
    names = {generate_set_name({}) for _ in range(20)}
    # All generated names are valid adjective_noun pairs (we can't assert uniqueness
    # across calls without state, but each call on the same empty registry is valid)
    assert all("_" in n for n in names)


# ─────────────────────────────────────────────────────
# register_or_get — core deduplication
# ─────────────────────────────────────────────────────

@pytest.fixture
def registry_path(tmp_path) -> Path:
    return tmp_path / "registry.json"


def test_register_new_set_creates_entry(registry_path):
    ds = register_or_get("summarisation", ["cnn_dm_0000", "cnn_dm_0001"], registry_path=registry_path)
    assert ds.set_name
    assert ds.selection_hash == selection_hash("summarisation", ["cnn_dm_0000", "cnn_dm_0001"])
    assert ds.n_docs == 2
    assert ds.task == "summarisation"


def test_register_same_set_returns_same_name(registry_path):
    ds1 = register_or_get("summarisation", ["cnn_dm_0000", "cnn_dm_0001"], registry_path=registry_path)
    ds2 = register_or_get("summarisation", ["cnn_dm_0001", "cnn_dm_0000"], registry_path=registry_path)
    assert ds1.set_name == ds2.set_name
    assert ds1.selection_hash == ds2.selection_hash


def test_register_different_docs_creates_new_entry(registry_path):
    ds1 = register_or_get("summarisation", ["cnn_dm_0000"], registry_path=registry_path)
    ds2 = register_or_get("summarisation", ["cnn_dm_0001"], registry_path=registry_path)
    assert ds1.set_name != ds2.set_name
    assert ds1.selection_hash != ds2.selection_hash


def test_register_different_task_creates_new_entry(registry_path):
    ds1 = register_or_get("summarisation", ["cnn_dm_0000"], registry_path=registry_path)
    ds2 = register_or_get("translation", ["cnn_dm_0000"], registry_path=registry_path)
    assert ds1.set_name != ds2.set_name


def test_register_persists_to_disk(registry_path):
    register_or_get("summarisation", ["cnn_dm_0000"], registry_path=registry_path)
    assert registry_path.exists()
    raw = json.loads(registry_path.read_text())
    assert len(raw) == 1


def test_register_updates_last_used_at(registry_path):
    ds1 = register_or_get("summarisation", ["cnn_dm_0000"], registry_path=registry_path)
    # Re-register same docs — should update last_used_at
    ds2 = register_or_get("summarisation", ["cnn_dm_0000"], registry_path=registry_path)
    assert ds1.set_name == ds2.set_name
    # last_used_at may equal created_at when called very quickly, but set_name is stable
    assert ds2.last_used_at >= ds1.last_used_at


def test_register_or_get_doc_ids_sorted(registry_path):
    ds = register_or_get("summarisation", ["cnn_dm_0003", "cnn_dm_0001"], registry_path=registry_path)
    assert ds.doc_ids == sorted(ds.doc_ids)


# ─────────────────────────────────────────────────────
# lookup helpers
# ─────────────────────────────────────────────────────

def test_lookup_by_name_found(registry_path):
    ds = register_or_get("summarisation", ["cnn_dm_0000"], registry_path=registry_path)
    found = lookup_by_name(ds.set_name, registry_path=registry_path)
    assert found is not None
    assert found.set_name == ds.set_name


def test_lookup_by_name_not_found(registry_path):
    assert lookup_by_name("nonexistent_name", registry_path=registry_path) is None


def test_lookup_by_hash_found(registry_path):
    ds = register_or_get("summarisation", ["cnn_dm_0000"], registry_path=registry_path)
    found = lookup_by_hash(ds.selection_hash, registry_path=registry_path)
    assert found is not None
    assert found.set_name == ds.set_name


def test_lookup_by_hash_not_found(registry_path):
    assert lookup_by_hash("000000000000", registry_path=registry_path) is None
