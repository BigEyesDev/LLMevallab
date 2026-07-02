"""Token usage and cost calculation for LLM API calls."""

from pydantic import BaseModel, computed_field


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    @computed_field
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def calculate_cost(usage: TokenUsage, pricing: dict) -> float:
    """Compute USD cost from token counts and per-million pricing rates."""
    try:
        input_per_1m = pricing["input_per_1m"]
        output_per_1m = pricing["output_per_1m"]
    except KeyError as exc:
        raise ValueError(f"pricing dict missing required key: {exc.args[0]}") from exc

    input_cost = usage.input_tokens * input_per_1m / 1_000_000
    output_cost = usage.output_tokens * output_per_1m / 1_000_000
    return round(input_cost + output_cost, 8)
