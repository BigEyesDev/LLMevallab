"""LLMevallab — model-agnostic document evaluation pipeline."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("llmevallab")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
