from src.core.time import utc_now, utc_now_iso, utc_timestamp


def test_utc_now_is_timezone_aware():
    assert utc_now().tzinfo is not None


def test_utc_now_iso_is_parseable():
    assert "T" in utc_now_iso()


def test_utc_timestamp_format():
    assert len(utc_timestamp()) == 15
    assert utc_timestamp()[8] == "_"
