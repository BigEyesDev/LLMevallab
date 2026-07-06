"""Load, version, and persist LLM prompt templates."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from src.core.time import utc_timestamp

DEFAULT_PROMPTS_PATH = Path("configs/prompts.yaml")
PROMPT_HISTORY_DIR = Path("configs/prompt_history")

_TASK_KEYS = ("extraction", "translation", "summarisation")


def load_prompts(prompts_path: str | Path = DEFAULT_PROMPTS_PATH) -> dict:
    """Load prompts.yaml; returns empty dict if file is missing or empty."""
    path = Path(prompts_path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def get_prompt_version(prompts: dict) -> str:
    """Return the active prompt version string (defaults to '1' when absent)."""
    return str(prompts.get("version", 1))


def get_task_prompts(prompts: dict, task: str) -> dict[str, str]:
    """Return system and user template strings for a pipeline task."""
    task_data = prompts.get(task, {}) or {}
    return {
        "system": task_data.get("system", ""),
        "user": task_data.get("user", ""),
    }


def _slugify_note(note: str) -> str:
    slug = re.sub(r"[^\w]+", "_", note.strip().lower()).strip("_")
    return slug[:40] or "edit"


def _prompt_tasks_only(prompts: dict) -> dict:
    """Strip the top-level version key — task blocks only."""
    return {k: v for k, v in prompts.items() if k in _TASK_KEYS}


def save_prompt_version(
    task: str,
    system: str,
    user: str,
    note: str,
    *,
    prompts_path: str | Path = DEFAULT_PROMPTS_PATH,
    history_dir: str | Path = PROMPT_HISTORY_DIR,
) -> str:
    """Snapshot current prompts, update one task, increment version. Returns new version."""
    if task not in _TASK_KEYS:
        raise ValueError(f"Unknown prompt task: {task!r}. Expected one of {_TASK_KEYS}.")

    path = Path(prompts_path)
    history = Path(history_dir)
    history.mkdir(parents=True, exist_ok=True)

    current = load_prompts(path)
    old_version = int(current.get("version", 1))
    ts = utc_timestamp()
    slug = _slugify_note(note)
    snapshot_path = history / f"v{old_version}_{ts}_{slug}.yaml"

    snapshot = {
        "version": old_version,
        "timestamp": ts,
        "note": note.strip() or "snapshot",
        "prompts": _prompt_tasks_only(current),
    }
    with open(snapshot_path, "w", encoding="utf-8") as f:
        yaml.dump(snapshot, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    updated = dict(current)
    updated["version"] = old_version + 1
    updated[task] = {"system": system, "user": user}

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(updated, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    return str(old_version + 1)
