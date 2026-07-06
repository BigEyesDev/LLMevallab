"""Tests for src/pipeline/prompt_manager.py."""

import yaml

import pytest

from src.pipeline.prompt_manager import (
    get_prompt_version,
    get_task_prompts,
    load_prompts,
    save_prompt_version,
)


def _write_prompts(path, content: dict) -> None:
    path.write_text(yaml.dump(content, default_flow_style=False), encoding="utf-8")


def test_load_prompts_returns_version(tmp_path):
    prompts_file = tmp_path / "prompts.yaml"
    _write_prompts(prompts_file, {"version": 2, "translation": {"system": "s", "user": "u"}})

    data = load_prompts(prompts_file)
    assert data["version"] == 2


def test_get_prompt_version_defaults_to_one():
    assert get_prompt_version({}) == "1"
    assert get_prompt_version({"version": 3}) == "3"


def test_get_task_prompts_returns_system_and_user():
    prompts = {
        "version": 1,
        "summarisation": {"system": "sys", "user": "usr {text}"},
    }
    task = get_task_prompts(prompts, "summarisation")
    assert task["system"] == "sys"
    assert task["user"] == "usr {text}"


def test_get_task_prompts_missing_task_returns_empty_strings():
    assert get_task_prompts({}, "translation") == {"system": "", "user": ""}


def test_save_prompt_version_increments_and_snapshots(tmp_path):
    prompts_file = tmp_path / "prompts.yaml"
    history_dir = tmp_path / "history"
    initial = {
        "version": 1,
        "translation": {"system": "old sys", "user": "old usr"},
        "summarisation": {"system": "sum sys", "user": "sum usr"},
    }
    _write_prompts(prompts_file, initial)

    new_version = save_prompt_version(
        "translation",
        "new sys",
        "new usr",
        "shorter translation",
        prompts_path=prompts_file,
        history_dir=history_dir,
    )

    assert new_version == "2"
    updated = load_prompts(prompts_file)
    assert updated["version"] == 2
    assert updated["translation"]["system"] == "new sys"
    assert updated["summarisation"]["system"] == "sum sys"

    snapshots = list(history_dir.glob("v1_*.yaml"))
    assert len(snapshots) == 1
    snapshot = yaml.safe_load(snapshots[0].read_text(encoding="utf-8"))
    assert snapshot["version"] == 1
    assert snapshot["note"] == "shorter translation"
    assert snapshot["prompts"]["translation"]["system"] == "old sys"


def test_save_prompt_version_unknown_task_raises(tmp_path):
    prompts_file = tmp_path / "prompts.yaml"
    _write_prompts(prompts_file, {"version": 1, "translation": {"system": "s", "user": "u"}})

    with pytest.raises(ValueError, match="Unknown prompt task"):
        save_prompt_version(
            "unknown",
            "s",
            "u",
            "note",
            prompts_path=prompts_file,
            history_dir=tmp_path / "history",
        )
