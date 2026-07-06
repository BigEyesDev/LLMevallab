"""Lightweight LLM judge client for evaluation metrics."""

import json
import logging
import os
import time

import google.generativeai as genai
from anthropic import Anthropic
from openai import OpenAI

from src.core.config import get_model_catalog, validate_model_key
from src.core.pricing import TokenUsage, calculate_cost

logger = logging.getLogger(__name__)

JUDGE_SYSTEM = (
    "You are an expert evaluator of text summaries. "
    "Respond with valid JSON only — no markdown fences."
)

JUDGE_USER_TEMPLATE = """Rate the GENERATED SUMMARY against the SOURCE DOCUMENT on three dimensions (integer 1-5):

1. **faithfulness** — accurate to the source, no hallucinations
2. **completeness** — captures the key information from the source
3. **coherence** — well-written, fluent, and logically structured

Return ONLY this JSON object:
{{"faithfulness": <1-5>, "completeness": <1-5>, "coherence": <1-5>, "reasoning": "<brief explanation>"}}

SOURCE DOCUMENT:
{source}

GENERATED SUMMARY:
{hypothesis}"""


class JudgeClient:
    """Calls a configured catalog model to score summary quality."""

    def __init__(self, config: dict, model_key: str):
        validate_model_key(model_key, config)
        self.model_key = model_key
        self.model_config = get_model_catalog(config)[model_key]
        self.provider_type = self.model_config["provider_type"]
        api_key = os.environ.get(self.model_config["api_key_env"])
        if not api_key:
            raise EnvironmentError(
                f"{self.model_config['api_key_env']} not set — required for LLM judge."
            )

        if self.provider_type == "gemini":
            genai.configure(api_key=api_key)
            self._gemini = genai.GenerativeModel(
                model_name=self.model_config["model_id"],
                generation_config=genai.GenerationConfig(
                    temperature=self.model_config.get("temperature", 0.1),
                    max_output_tokens=self.model_config.get("max_output_tokens", 512),
                ),
            )
        elif self.provider_type == "claude":
            self._claude = Anthropic(api_key=api_key)
        elif self.provider_type == "openai_compatible":
            self._openai = OpenAI(
                api_key=api_key,
                base_url=self.model_config.get("base_url"),
            )
        else:
            raise ValueError(f"Unsupported judge provider_type: '{self.provider_type}'")

    def evaluate(self, source: str, hypothesis: str) -> tuple[dict, float, dict]:
        """
        Score one summary.

        Returns:
            (scores_dict, latency_ms, metadata) where scores_dict has
            faithfulness/completeness/coherence (1-5) and optional reasoning.
        """
        user = JUDGE_USER_TEMPLATE.format(source=source, hypothesis=hypothesis)
        start = time.time()

        if self.provider_type == "gemini":
            raw, usage = self._call_gemini(user)
        elif self.provider_type == "claude":
            raw, usage = self._call_claude(user)
        else:
            raw, usage = self._call_openai(user)

        latency_ms = (time.time() - start) * 1000
        scores = _parse_judge_json(raw)
        pricing = self.model_config.get("pricing", {})
        cost_usd = calculate_cost(usage, pricing) if pricing else 0.0

        metadata = {
            "judge_model": self.model_config["model_id"],
            "judge_model_key": self.model_key,
            "latency_ms": round(latency_ms, 2),
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cost_usd": cost_usd,
            "reasoning": scores.pop("reasoning", ""),
        }
        return scores, latency_ms, metadata

    def _call_gemini(self, user: str) -> tuple[str, TokenUsage]:
        prompt = f"{JUDGE_SYSTEM}\n\n{user}"
        response = self._gemini.generate_content(prompt)
        metadata = getattr(response, "usage_metadata", None)
        usage = TokenUsage(
            input_tokens=getattr(metadata, "prompt_token_count", 0) or 0,
            output_tokens=getattr(metadata, "candidates_token_count", 0) or 0,
        )
        return response.text, usage

    def _call_claude(self, user: str) -> tuple[str, TokenUsage]:
        response = self._claude.messages.create(
            model=self.model_config["model_id"],
            max_tokens=self.model_config.get("max_output_tokens", 512),
            temperature=self.model_config.get("temperature", 0.1),
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return response.content[0].text, usage

    def _call_openai(self, user: str) -> tuple[str, TokenUsage]:
        response = self._openai.chat.completions.create(
            model=self.model_config["model_id"],
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=self.model_config.get("temperature", 0.1),
            max_tokens=self.model_config.get("max_output_tokens", 512),
        )
        usage = TokenUsage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )
        text = response.choices[0].message.content or ""
        return text, usage


def _parse_judge_json(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning(f"Failed to parse judge JSON: {exc}. Raw: {raw[:200]}")
        return {"faithfulness": 1, "completeness": 1, "coherence": 1, "reasoning": "parse_error"}

    for key in ("faithfulness", "completeness", "coherence"):
        if key not in data:
            data[key] = 1
        else:
            data[key] = max(1, min(5, int(data[key])))
    return data


def normalize_score(value: int) -> float:
    """Map 1-5 integer rating to 0-1 scale."""
    return round((value - 1) / 4, 4)
