"""Document Set Registry — named, content-addressed benchmark slices.

Each unique (task, doc_ids) combination gets a stable human name such as
``pearl_wish`` and a 12-char content hash.  The registry lives at
``data/document_sets/registry.json`` and is git-trackable.

Typical call site (BenchmarkRunner.run):
    from src.pipeline.document_sets import register_or_get
    doc_set = register_or_get(task, doc_ids)
    # → DocumentSet(set_name='pearl_wish', selection_hash='a3f9c2e1b847', ...)
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from src.core.provenance import selection_hash as _compute_selection_hash
from src.core.time import utc_now_iso

_REGISTRY_PATH = Path("data/document_sets/registry.json")

# ─── Word lists for auto-generated set names ───────────────────────────────
_ADJECTIVES = [
    "amber", "azure", "bright", "calm", "cedar", "clear", "coral",
    "crisp", "dawn", "deep", "dusk", "fern", "firm", "fresh", "frost",
    "golden", "green", "grey", "jade", "keen", "lake", "lemon", "lunar",
    "maple", "mild", "mint", "mist", "moss", "muted", "mystic", "naval",
    "oak", "opal", "pale", "pearl", "pine", "plain", "polar", "prism",
    "quiet", "rapid", "rose", "ruby", "sage", "sandy", "serene", "silver",
    "slate", "snow", "soft", "solar", "stern", "stone", "storm", "sunlit",
    "swift", "teal", "terra", "tide", "timber", "true", "vast", "velvet",
    "warm", "wave", "white", "wild", "windy", "wise",
]

_NOUNS = [
    "arch", "bay", "beam", "bench", "bloom", "bold", "branch", "breeze",
    "bridge", "brook", "brush", "calm", "cave", "chord", "cliff", "cloud",
    "coast", "core", "craft", "creek", "crest", "drift", "dune", "edge",
    "fable", "field", "flame", "flare", "fleet", "flow", "fold", "forge",
    "glade", "gleam", "glow", "grant", "grove", "guild", "haven", "helm",
    "hill", "hold", "holt", "hope", "hue", "isle", "key", "knoll",
    "lake", "lane", "leaf", "ledge", "light", "loft", "mark", "marsh",
    "mesa", "mill", "mire", "mist", "moor", "peak", "pebble", "plain",
    "plume", "point", "pool", "port", "pulse", "quest", "reach", "reef",
    "ridge", "rift", "ring", "rise", "river", "road", "rock", "sail",
    "scope", "seed", "shade", "shore", "slope", "span", "spark", "spire",
    "spring", "spur", "stand", "star", "stem", "step", "stone", "stream",
    "surge", "swept", "tide", "trail", "vale", "vault", "veil", "view",
    "wake", "ward", "wave", "well", "wind", "wish", "wood", "yard",
]


# ─── Domain model ──────────────────────────────────────────────────────────

@dataclass
class DocumentSet:
    """One named, content-addressed benchmark slice."""

    set_name: str
    selection_hash: str
    task: str
    doc_ids: list[str]
    data_source: str
    n_docs: int
    created_at: str
    last_used_at: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DocumentSet":
        return cls(**d)


# ─── Registry I/O ──────────────────────────────────────────────────────────

def load_registry(registry_path: Path = _REGISTRY_PATH) -> dict[str, DocumentSet]:
    """Load the on-disk registry.  Returns ``{}`` when the file does not exist."""
    if not registry_path.exists():
        return {}
    with open(registry_path, "r", encoding="utf-8") as f:
        raw: dict = json.load(f)
    return {name: DocumentSet.from_dict(entry) for name, entry in raw.items()}


def save_registry(
    registry: dict[str, DocumentSet],
    registry_path: Path = _REGISTRY_PATH,
) -> None:
    """Persist the registry to disk, creating parent dirs if needed."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(
            {name: ds.to_dict() for name, ds in registry.items()},
            f,
            ensure_ascii=False,
            indent=2,
        )


# ─── Name generation ───────────────────────────────────────────────────────

def generate_set_name(registry: dict[str, DocumentSet]) -> str:
    """Return a unique ``{adjective}_{noun}`` name not already in ``registry``.

    Makes up to 500 random attempts before falling back to a counter suffix.
    """
    existing = set(registry.keys())
    for _ in range(500):
        name = f"{random.choice(_ADJECTIVES)}_{random.choice(_NOUNS)}"
        if name not in existing:
            return name
    base = f"{random.choice(_ADJECTIVES)}_{random.choice(_NOUNS)}"
    i = 2
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


# ─── Core API ──────────────────────────────────────────────────────────────

def register_or_get(
    task: str,
    doc_ids: list[str],
    data_source: str = "benchmark_samples",
    registry_path: Path = _REGISTRY_PATH,
) -> DocumentSet:
    """Return the DocumentSet for this (task, doc_ids) pair; create one if new.

    Deduplication is by content hash — the same 10 docs on any day return the
    same set name.  ``last_used_at`` is always updated on hit.

    Args:
        task:          Pipeline task string (e.g. ``'summarisation'``).
        doc_ids:       Document IDs selected for this run.
        data_source:   Origin label stored in the set record.
        registry_path: Override the default registry file path (useful in tests).

    Returns:
        The (possibly newly created) :class:`DocumentSet`.
    """
    registry = load_registry(registry_path)
    sel_hash = _compute_selection_hash(task, doc_ids)
    now = utc_now_iso()

    for ds in registry.values():
        if ds.selection_hash == sel_hash:
            ds.last_used_at = now
            save_registry(registry, registry_path)
            return ds

    name = generate_set_name(registry)
    ds = DocumentSet(
        set_name=name,
        selection_hash=sel_hash,
        task=task,
        doc_ids=sorted(doc_ids),
        data_source=data_source,
        n_docs=len(doc_ids),
        created_at=now,
        last_used_at=now,
    )
    registry[name] = ds
    save_registry(registry, registry_path)
    return ds


def lookup_by_name(
    name: str,
    registry_path: Path = _REGISTRY_PATH,
) -> DocumentSet | None:
    """Return the DocumentSet with this name, or ``None``."""
    return load_registry(registry_path).get(name)


def lookup_by_hash(
    sel_hash: str,
    registry_path: Path = _REGISTRY_PATH,
) -> DocumentSet | None:
    """Return the DocumentSet whose ``selection_hash`` matches, or ``None``."""
    for ds in load_registry(registry_path).values():
        if ds.selection_hash == sel_hash:
            return ds
    return None
