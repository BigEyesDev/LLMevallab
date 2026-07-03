import pytest

from src.core.pricing import TokenUsage, calculate_cost


def test_calculate_cost_basic():
    usage = TokenUsage(input_tokens=1000, output_tokens=500)
    pricing = {"input_per_1m": 0.075, "output_per_1m": 0.30}
    cost = calculate_cost(usage, pricing)
    expected = (1000 * 0.075 / 1_000_000) + (500 * 0.30 / 1_000_000)
    assert cost == pytest.approx(expected)


def test_calculate_cost_zero_tokens():
    usage = TokenUsage(input_tokens=0, output_tokens=0)
    pricing = {"input_per_1m": 0.075, "output_per_1m": 0.30}
    assert calculate_cost(usage, pricing) == 0.0


def test_calculate_cost_missing_pricing_raises():
    usage = TokenUsage(input_tokens=100, output_tokens=50)
    with pytest.raises(ValueError, match="missing required key"):
        calculate_cost(usage, {"input_per_1m": 0.1})


def test_token_usage_total_tokens():
    usage = TokenUsage(input_tokens=800, output_tokens=200)
    assert usage.total_tokens == 1000
