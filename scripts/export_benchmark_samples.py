"""
One-time export script: reads processed JSON files and writes curated benchmark
samples to data/benchmark_samples/.

Run once after `python main.py` has downloaded the full dataset:

    uv run python scripts/export_benchmark_samples.py

Output files are committed to git so the dashboard works offline with no setup.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
OUTPUT_DIR = REPO_ROOT / "data" / "benchmark_samples"

SOURCES: dict[str, dict] = {
    "translation": {
        "candidates": [
            PROCESSED_DIR / "europarl" / "europarl_de-en_20docs.json",
            PROCESSED_DIR / "europarl_de-en_20docs.json",  # legacy flat location
        ],
        "output": OUTPUT_DIR / "translation_de_en.json",
        "hf_dataset": "Helsinki-NLP/europarl",
        "description": "EuroParl DE→EN parallel sentences, 30 docs",
    },
    "summarisation": {
        "candidates": [
            PROCESSED_DIR / "cnn_dailymail" / "cnn_dailymail_20docs.json",
            PROCESSED_DIR / "cnn_dailymail" / "cnn_dailymail_7docs.json",
        ],
        "output": OUTPUT_DIR / "summarisation_en.json",
        "hf_dataset": "abisee/cnn_dailymail (3.0.0, test split)",
        "description": "CNN/DailyMail news articles + reference summaries, 30 docs",
    },
}

MAX_DOCS = 30


def _resolve_source(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def export_task(task: str, spec: dict) -> int:
    """Export up to MAX_DOCS from the best available processed file. Returns doc count."""
    source_path = _resolve_source(spec["candidates"])
    if source_path is None:
        searched = ", ".join(str(p) for p in spec["candidates"])
        print(f"  [SKIP] {task}: no processed file found. Searched: {searched}")
        print(f"         Run `python main.py` first to download the dataset.")
        return 0

    raw: list[dict] = json.loads(source_path.read_text(encoding="utf-8"))
    docs = raw[:MAX_DOCS]

    output_path: Path = spec["output"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  [OK]   {task}: wrote {len(docs)} docs → {output_path.relative_to(REPO_ROOT)}")
    print(f"         source: {source_path.relative_to(REPO_ROOT)}")
    return len(docs)


def write_readme(exported: dict[str, int]) -> None:
    """Write provenance README alongside the benchmark samples."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    doc_ids: dict[str, list[str]] = {}
    for task, spec in SOURCES.items():
        output_path: Path = spec["output"]
        if output_path.exists() and exported.get(task, 0) > 0:
            data = json.loads(output_path.read_text(encoding="utf-8"))
            doc_ids[task] = [d["doc_id"] for d in data]

    lines = [
        "# Benchmark Samples — Provenance",
        "",
        f"Generated: {timestamp}",
        "",
        "## Files",
        "",
        "| File | Task | Source Dataset | Docs |",
        "|---|---|---|---|",
    ]

    for task, spec in SOURCES.items():
        n = exported.get(task, 0)
        fname = spec["output"].name
        lines.append(f"| `{fname}` | {task} | {spec['hf_dataset']} | {n} |")

    lines += ["", "## Document IDs", ""]
    for task, ids in doc_ids.items():
        lines.append(f"### {task}")
        lines.append("")
        lines.append(", ".join(f"`{i}`" for i in ids))
        lines.append("")

    lines += [
        "## Schema",
        "",
        "**translation_de_en.json** — each entry:",
        "```json",
        '{',
        '  "doc_id": "europarl_de-en_0000",',
        '  "source_language": "de",',
        '  "raw_text": "...",',
        '  "source": "europarl",',
        '  "metadata": {"reference_translation": "..."}',
        '}',
        "```",
        "",
        "**summarisation_en.json** — each entry:",
        "```json",
        '{',
        '  "doc_id": "cnn_dm_0000",',
        '  "source_language": "en",',
        '  "raw_text": "...",',
        '  "source": "cnn_dailymail",',
        '  "metadata": {"reference_summary": "...", "cnn_id": "..."}',
        '}',
        "```",
        "",
        "These files are committed to git. "
        "Re-run this script after `python main.py` downloads larger samples.",
    ]

    readme_path = OUTPUT_DIR / "README.md"
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  [OK]   README.md → {readme_path.relative_to(REPO_ROOT)}")


def main() -> None:
    print("Exporting benchmark samples…")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    exported: dict[str, int] = {}
    for task, spec in SOURCES.items():
        exported[task] = export_task(task, spec)

    write_readme(exported)

    total = sum(exported.values())
    if total == 0:
        print("\nNo samples exported. Run `python main.py` first.")
        sys.exit(1)

    print(f"\nDone — {total} documents exported to {OUTPUT_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
