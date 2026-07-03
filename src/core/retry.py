"""Retry decorator for transient LLM API failures."""

from tenacity import retry, stop_after_attempt, wait_exponential


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Retry a function with exponential backoff on any exception."""
    return retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=base_delay, min=base_delay),
        reraise=True,
    )
