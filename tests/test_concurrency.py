import threading
import time

from src.core.concurrency import ProviderLimiter, run_in_parallel
from src.core.config import get_concurrency_settings, load_config


def test_run_in_parallel_preserves_order():
    items = list(range(10))
    results = run_in_parallel(items, lambda x: x * 2, max_workers=4)
    assert results == [x * 2 for x in items]


def test_run_in_parallel_max_workers_one_is_sequential():
    counter = {"n": 0}

    def worker(x):
        counter["n"] += 1
        return x

    results = run_in_parallel([1, 2, 3], worker, max_workers=1)
    assert results == [1, 2, 3]
    assert counter["n"] == 3


def test_run_in_parallel_respects_max_workers():
    lock = threading.Lock()
    active = {"current": 0, "peak": 0}

    def worker(_):
        with lock:
            active["current"] += 1
            active["peak"] = max(active["peak"], active["current"])
        time.sleep(0.05)
        with lock:
            active["current"] -= 1
        return True

    run_in_parallel(list(range(8)), worker, max_workers=2)
    assert active["peak"] <= 2


def test_provider_limiter_blocks_when_exhausted():
    limiter = ProviderLimiter({"gemini": 1})
    sem = limiter.acquire("gemini")
    acquired = threading.Event()
    released = threading.Event()

    def holder():
        with sem:
            acquired.set()
            released.wait(timeout=2)

    t = threading.Thread(target=holder)
    t.start()
    acquired.wait(timeout=2)

    blocked = threading.Event()

    def waiter():
        with sem:
            blocked.set()

    t2 = threading.Thread(target=waiter)
    t2.start()
    time.sleep(0.05)
    assert not blocked.is_set()
    released.set()
    blocked.wait(timeout=2)
    t.join(timeout=2)
    t2.join(timeout=2)
    assert blocked.is_set()


def test_get_concurrency_settings_reads_config():
    config = load_config()
    settings = get_concurrency_settings(config)
    assert settings.max_concurrent_documents == 5
    assert settings.max_concurrent_models == 3
    assert settings.max_concurrent_judge_calls == 5
    assert settings.provider_limits["gemini"] == 10
