import pytest

from src.core.retry import retry_with_backoff


def test_retry_succeeds_after_transient_failures():
    calls = {"count": 0}

    @retry_with_backoff(max_retries=3, base_delay=0.01)
    def flaky():
        calls["count"] += 1
        if calls["count"] < 3:
            raise RuntimeError("transient")
        return "ok"

    assert flaky() == "ok"
    assert calls["count"] == 3


def test_retry_gives_up_after_max_retries():
    calls = {"count": 0}

    @retry_with_backoff(max_retries=3, base_delay=0.01)
    def always_fails():
        calls["count"] += 1
        raise RuntimeError("permanent")

    with pytest.raises(RuntimeError, match="permanent"):
        always_fails()
    assert calls["count"] == 3


def test_retry_no_extra_calls_on_first_try_success():
    calls = {"count": 0}

    @retry_with_backoff(max_retries=3, base_delay=0.01)
    def succeeds():
        calls["count"] += 1
        return "done"

    assert succeeds() == "done"
    assert calls["count"] == 1
