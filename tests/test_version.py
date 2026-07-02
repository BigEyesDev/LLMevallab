import re

import pytest

from src import __version__


def test_package_version_is_semver_like():
    assert re.match(r"^\d+\.\d+\.\d+(\.dev\d+)?(\+[\w.]+)?$", __version__)


def test_package_version_matches_pyproject():
    from pathlib import Path

    text = Path("pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"', text, re.MULTILINE)
    assert match is not None
    assert __version__ == match.group(1)
