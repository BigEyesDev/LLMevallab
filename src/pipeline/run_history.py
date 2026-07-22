"""Load and group past benchmark runs from on-disk RunManifest files."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.core.models import RunManifest

# Manifests from the same multi-model dashboard run usually land within minutes;
# allow up to 2 hours for slow models before treating as a separate session.
SESSION_WINDOW_SECONDS = 7200

_RESULTS_STEM = re.compile(
    r"^results_(translation|summarisation|full)_(.+)_(\d{8}_\d{6})$"
)


@dataclass
class PastRunSession:
    """One benchmark session: same task, same doc set, one or more models."""

    task: str
    doc_ids: list[str]
    models: list[str] = field(default_factory=list)
    started_at: str = ""
    latest_at: str = ""
    prompt_version: str | None = None

    @property
    def n_docs(self) -> int:
        return len(self.doc_ids)


def _parse_ts(ts: str) -> datetime:
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _model_from_results_path(results_path: str) -> str:
    """Infer a display model slug from a results filename when manifest.model_key is empty."""
    stem = Path(results_path).stem
    match = _RESULTS_STEM.match(stem)
    if match:
        return match.group(2).replace("_", "-")
    return ""


def _resolve_model_key(manifest: RunManifest) -> str:
    if manifest.model_key:
        return manifest.model_key
    return _model_from_results_path(manifest.results_path)


def _prompt_from_manifest(manifest: RunManifest) -> str | None:
    snapshot = manifest.config_snapshot or {}
    prompts = snapshot.get("prompts") or snapshot.get("prompt_version")
    if isinstance(prompts, str):
        return prompts
    return None


def load_past_run_sessions(
    outputs_dir: Path | str,
    *,
    task: str | None = None,
    limit: int = 10,
) -> list[PastRunSession]:
    """Scan ``*.manifest.json`` files and group them into benchmark sessions.

    Sessions are grouped by matching ``task`` + ``doc_ids`` when manifests were
    written within ``SESSION_WINDOW_SECONDS`` of each other (typical multi-model
    dashboard run). Returns the most recent sessions first.
    """
    root = Path(outputs_dir)
    if not root.is_dir():
        return []

    manifests: list[RunManifest] = []
    for path in sorted(root.glob("*.manifest.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            manifest = RunManifest.model_validate(data)
        except (json.JSONDecodeError, ValueError, OSError):
            continue
        if not manifest.doc_ids:
            continue
        if task is not None and manifest.task != task:
            continue
        manifests.append(manifest)

    if not manifests:
        return []

    manifests.sort(key=lambda m: _parse_ts(m.created_at))

    sessions: list[PastRunSession] = []
    for manifest in manifests:
        doc_key = tuple(sorted(manifest.doc_ids))
        ts = _parse_ts(manifest.created_at)
        model = _resolve_model_key(manifest)

        matched: PastRunSession | None = None
        for session in reversed(sessions):
            if session.task != manifest.task:
                continue
            if tuple(sorted(session.doc_ids)) != doc_key:
                continue
            latest = _parse_ts(session.latest_at)
            if abs((ts - latest).total_seconds()) <= SESSION_WINDOW_SECONDS:
                matched = session
                break

        if matched is None:
            sessions.append(
                PastRunSession(
                    task=manifest.task,
                    doc_ids=list(manifest.doc_ids),
                    models=[model] if model else [],
                    started_at=manifest.created_at,
                    latest_at=manifest.created_at,
                    prompt_version=_prompt_from_manifest(manifest),
                )
            )
            continue

        if model and model not in matched.models:
            matched.models.append(model)
        if ts > _parse_ts(matched.latest_at):
            matched.latest_at = manifest.created_at
        if ts < _parse_ts(matched.started_at):
            matched.started_at = manifest.created_at

    sessions.sort(key=lambda s: _parse_ts(s.latest_at), reverse=True)
    return sessions[:limit]
