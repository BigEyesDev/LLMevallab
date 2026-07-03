"""UTC timestamps for run metadata and artifact filenames."""

from datetime import UTC, datetime


def utc_now() -> datetime:
    return datetime.now(UTC)


def utc_now_iso() -> str:
    return utc_now().isoformat()


def utc_timestamp() -> str:
    """Compact UTC timestamp for filenames, e.g. 20260703_113800."""
    return utc_now().strftime("%Y%m%d_%H%M%S")
