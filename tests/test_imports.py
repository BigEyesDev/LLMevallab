"""Smoke test: Phase 2 modules import cleanly."""


def test_import_pricing():
    from src.core.pricing import TokenUsage, calculate_cost

    assert TokenUsage(input_tokens=1, output_tokens=1).total_tokens == 2
    assert calculate_cost(
        TokenUsage(input_tokens=100, output_tokens=50),
        {"input_per_1m": 0.1, "output_per_1m": 0.1},
    ) >= 0


def test_import_retry():
    from src.core.retry import retry_with_backoff

    assert callable(retry_with_backoff)


def test_import_providers():
    from src.providers.claude_processor import ClaudeProcessor
    from src.providers.gemini_processor import GeminiProcessor
    from src.providers.openai_compatible_processor import OpenAICompatibleProcessor

    assert ClaudeProcessor and GeminiProcessor and OpenAICompatibleProcessor


def test_import_benchmark():
    from src.evaluations.benchmark import BenchmarkRunner, parse_models

    assert BenchmarkRunner and parse_models("a,b") == ["a", "b"]
