"""Loads committed static benchmark samples — no network or HF download required."""

from __future__ import annotations

import json
from pathlib import Path

from src.core.models import DocumentInput

_SAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "benchmark_samples"

_TASK_FILE: dict[str, str] = {
    "translation": "translation_de_en.json",
    "summarisation": "summarisation_en.json",
}

_TASK_GROUND_TRUTH_KEY: dict[str, str] = {
    "translation": "reference_translation",
    "summarisation": "reference_summary",
}


class BenchmarkSampleLoader:
    """
    Reads committed static JSON benchmark samples from data/benchmark_samples/.

    No network required — files are committed to git and always available.
    Follows the same interface as EuroParlDataLoader and CNNDailyMailLoader.
    """

    def __init__(self, samples_dir: str | Path | None = None) -> None:
        self._dir = Path(samples_dir) if samples_dir else _SAMPLES_DIR

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, task: str) -> list[DocumentInput]:
        """Return all benchmark documents for *task* as DocumentInput objects."""
        path = self._resolve_path(task)
        raw: list[dict] = json.loads(path.read_text(encoding="utf-8"))
        return [DocumentInput(**doc) for doc in raw]

    def ground_truth(self, task: str) -> dict[str, str]:
        """Return {doc_id: reference_text} ground-truth mapping for *task*."""
        gt_key = _TASK_GROUND_TRUTH_KEY.get(task)
        if gt_key is None:
            raise ValueError(f"No ground-truth key configured for task '{task}'")

        docs = self.load(task)
        return {
            doc.doc_id: doc.metadata.get(gt_key, "")
            for doc in docs
        }

    def available_tasks(self) -> list[str]:
        """Return tasks whose sample file exists on disk."""
        return [task for task in _TASK_FILE if self._resolve_path(task, raise_on_missing=False) is not None]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, task: str, raise_on_missing: bool = True) -> Path | None:
        filename = _TASK_FILE.get(task)
        if filename is None:
            if raise_on_missing:
                raise ValueError(
                    f"Unknown task '{task}'. "
                    f"Valid tasks: {sorted(_TASK_FILE.keys())}"
                )
            return None

        path = self._dir / filename
        if not path.exists():
            if raise_on_missing:
                raise FileNotFoundError(
                    f"Benchmark sample file not found: {path}. "
                    "Run `python scripts/export_benchmark_samples.py` to generate it."
                )
            return None
        return path
