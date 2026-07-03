"""Run provenance utilities — file hashing and config snapshotting for manifest integrity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_hash(path: str | Path) -> str:
    """SHA-256 hash of a file's raw bytes.

    Streams the file in 64 KiB chunks so large datasets don't spike memory.

    Returns:
        'sha256:<hex>' — prefixed so the algorithm is self-documenting in the manifest.
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return f"sha256:{sha256.hexdigest()}"


#: Hash of the full input documents file — semantic alias for file_hash.
dataset_hash = file_hash


def ground_truth_hash(doc_ids: list[str], path: str | Path, gt_key: str) -> str:
    """SHA-256 hash of the ground truth texts for exactly the doc_ids in this run.

    Builds a canonical JSON dict of ``{sorted_doc_id: reference_text}`` so that:

    * Order of ``doc_ids`` does not affect the hash.
    * Any change to a reference text changes the hash.
    * Swapping the ground truth file to one with the same doc_ids but different
      reference texts changes the hash (catches the "wrong file" bug).

    Args:
        doc_ids:  Document IDs processed in this run.
        path:     Processed dataset JSON file path.
        gt_key:   Metadata key holding the reference text;
                  ``'reference_translation'`` for EuroParl or
                  ``'reference_summary'`` for CNN/DailyMail.

    Returns:
        ``'sha256:<hex>'`` string.
    """
    with open(path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    id_set = set(doc_ids)
    doc_map: dict[str, str] = {}
    for doc in documents:
        doc_id = doc.get("doc_id", "")
        if doc_id in id_set:
            doc_map[doc_id] = doc.get("metadata", {}).get(gt_key, "")

    canonical = json.dumps(
        {doc_id: doc_map.get(doc_id, "") for doc_id in sorted(doc_ids)},
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def config_hash(config: dict, model_key: str) -> str:
    """SHA-256 hash of the pipeline config block + the model's catalog entry.

    Captures the configuration state that directly influences inference so that
    a config change is visible in the manifest without re-reading every file.

    Args:
        config:    Loaded ``config.yaml`` dict.
        model_key: Catalog key (e.g. ``'gemini-2.5-flash'``).

    Returns:
        ``'sha256:<hex>'`` string.
    """
    snapshot = config_snapshot(config, model_key)
    canonical = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def config_snapshot(config: dict, model_key: str) -> dict:
    """Returns the task-relevant config slice stored in the manifest.

    Contains:
    * ``pipeline`` block — target language, max document length, etc.
    * ``model`` block — the full catalog entry for the model used.
    """
    return {
        "pipeline": config.get("pipeline", {}),
        "model": config.get("models", {}).get("catalog", {}).get(model_key, {}),
    }
