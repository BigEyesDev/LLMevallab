"""Lightweight evaluation settings — no orchestrator or metric imports."""


def get_judge_model_key(config: dict) -> str:
    """Catalog key for the LLM-as-Judge model."""
    return config.get("evaluation", {}).get("judge_model", "gpt-4o-mini")


def apply_judge_model_override(config: dict, judge_model_key: str) -> dict:
    """Return a config copy with evaluation.judge_model set."""
    updated = dict(config)
    evaluation = dict(config.get("evaluation", {}))
    evaluation["judge_model"] = judge_model_key
    updated["evaluation"] = evaluation
    return updated
