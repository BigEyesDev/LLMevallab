"""Bounded parallelism for I/O-bound API orchestration."""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")

DEFAULT_PROVIDER_LIMITS = {
    "gemini": 10,
    "claude": 5,
    "openai_compatible": 8,
}


@dataclass(frozen=True)
class ConcurrencySettings:
    """Concurrency knobs parsed from config.yaml."""

    max_concurrent_documents: int = 1
    skip_extraction: bool = False
    max_concurrent_models: int = 1
    max_concurrent_judge_calls: int = 1
    provider_limits: dict[str, int] | None = None

    def effective_provider_limits(self) -> dict[str, int]:
        return {**DEFAULT_PROVIDER_LIMITS, **(self.provider_limits or {})}


class ProviderLimiter:
    """Per-provider semaphores for API backpressure."""

    def __init__(self, limits: dict[str, int] | None = None):
        effective = {**DEFAULT_PROVIDER_LIMITS, **(limits or {})}
        self._semaphores = {
            provider: threading.Semaphore(max(1, limit))
            for provider, limit in effective.items()
        }

    def acquire(self, provider_type: str) -> threading.Semaphore:
        """Return the semaphore for provider_type (unlimited if unknown)."""
        if provider_type not in self._semaphores:
            self._semaphores[provider_type] = threading.Semaphore(10_000)
        return self._semaphores[provider_type]


def run_in_parallel(
    items: list[T],
    worker_fn: Callable[[T], R],
    max_workers: int,
    *,
    preserve_order: bool = True,
    on_complete: Callable[[T, R], None] | None = None,
) -> list[R]:
    """Run worker_fn over items with a bounded thread pool."""
    if not items:
        return []

    workers = max(1, min(max_workers, len(items)))

    if workers == 1:
        results = [worker_fn(item) for item in items]
        if on_complete:
            for item, result in zip(items, results):
                on_complete(item, result)
        return results

    indexed = list(enumerate(items))
    collected: list[tuple[int, R]] = []

    def _indexed_worker(pair: tuple[int, T]) -> tuple[int, R]:
        idx, item = pair
        result = worker_fn(item)
        if on_complete:
            on_complete(item, result)
        return idx, result

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for idx, result in executor.map(_indexed_worker, indexed):
            collected.append((idx, result))

    if preserve_order:
        collected.sort(key=lambda pair: pair[0])
    return [result for _, result in collected]
