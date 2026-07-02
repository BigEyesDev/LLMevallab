# src/providers/gemini_processor.py

import json
import logging
import time
from typing import Any

import google.generativeai as genai

from src.core.base_processor import BaseDocumentProcessor
from src.core.models import DocumentInput, ExtractionResult, TranslationResult, SummaryResult
from src.core.pricing import TokenUsage, calculate_cost
from src.core.retry import retry_with_backoff

logger = logging.getLogger(__name__)


class GeminiProcessor(BaseDocumentProcessor):
    """Gemini implementation of BaseDocumentProcessor."""

    def __init__(self, api_key: str, config: dict, prompts: dict):
        super().__init__(model_name=config["model_id"], config=config)

        genai.configure(api_key=api_key)

        generation_config = genai.GenerationConfig(
            temperature=config.get("temperature", 0.1),
            max_output_tokens=config.get("max_output_tokens", 2048),
        )

        self._client = genai.GenerativeModel(
            model_name=config["model_id"],
            generation_config=generation_config,
        )
        self._prompts = prompts
        self._timeout = config.get("timeout_seconds", 30)

    def extract(self, document: DocumentInput) -> ExtractionResult:
        """Extract entities, dates, deadlines, topics, and key clauses."""
        start = time.time()

        prompt = self._build_prompt(
            step="extraction",
            variables={"text": self._truncate(document.raw_text)},
        )
        raw_output, token_usage = self._call_api(prompt)
        parsed = self._parse_json(raw_output, step="extraction")
        cost_usd = self._cost_for_usage(token_usage)

        return ExtractionResult(
            doc_id=document.doc_id,
            entities=parsed.get("entities", []),
            dates=parsed.get("dates", []),
            deadlines=parsed.get("deadlines", []),
            topics=parsed.get("topics", []),
            key_clauses=parsed.get("key_clauses", []),
            raw_llm_output=raw_output,
            model_used=self.model_name,
            processing_time_ms=(time.time() - start) * 1000,
            token_usage=token_usage,
            cost_usd=cost_usd,
        )

    def translate(self, document: DocumentInput, target_language: str = "en") -> TranslationResult:
        """Translate document text to target language."""
        start = time.time()

        if document.source_language == target_language:
            logger.info(f"[{document.doc_id}] Already in {target_language}, skipping translation.")
            return TranslationResult(
                doc_id=document.doc_id,
                source_language=document.source_language,
                target_language=target_language,
                original_text=document.raw_text,
                translated_text=document.raw_text,
                model_used=self.model_name,
            )

        prompt = self._build_prompt(
            step="translation",
            variables={
                "text": self._truncate(document.raw_text),
                "source_language": document.source_language,
            },
        )
        translated_text, token_usage = self._call_api(prompt)
        cost_usd = self._cost_for_usage(token_usage)

        return TranslationResult(
            doc_id=document.doc_id,
            source_language=document.source_language,
            target_language=target_language,
            original_text=document.raw_text,
            translated_text=translated_text.strip(),
            model_used=self.model_name,
            processing_time_ms=(time.time() - start) * 1000,
            token_usage=token_usage,
            cost_usd=cost_usd,
        )

    def summarise(self, document: DocumentInput) -> SummaryResult:
        """Produce a structured English summary."""
        start = time.time()

        prompt = self._build_prompt(
            step="summarisation",
            variables={"text": self._truncate(document.raw_text)},
        )
        raw_output, token_usage = self._call_api(prompt)
        parsed = self._parse_json(raw_output, step="summarisation")
        cost_usd = self._cost_for_usage(token_usage)

        return SummaryResult(
            doc_id=document.doc_id,
            summary=parsed.get("summary", ""),
            key_points=parsed.get("key_points", []),
            action_items=parsed.get("action_items", []),
            model_used=self.model_name,
            processing_time_ms=(time.time() - start) * 1000,
            token_usage=token_usage,
            cost_usd=cost_usd,
        )

    def _build_prompt(self, step: str, variables: dict[str, str]) -> str:
        """Build a combined system + user prompt for Gemini."""
        step_prompts = self._prompts[step]
        system = step_prompts["system"]
        user = step_prompts["user"].format(**variables)
        return f"{system}\n\n{user}"

    @retry_with_backoff(max_retries=3, base_delay=0.01)
    def _call_api(self, prompt: str) -> tuple[str, TokenUsage]:
        """Call Gemini and return response text with token usage."""
        try:
            response = self._client.generate_content(prompt)
            metadata = getattr(response, "usage_metadata", None)
            usage = TokenUsage(
                input_tokens=getattr(metadata, "prompt_token_count", 0) or 0,
                output_tokens=getattr(metadata, "candidates_token_count", 0) or 0,
            )
            return response.text, usage
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise RuntimeError(f"Gemini API error: {e}") from e

    def _cost_for_usage(self, usage: TokenUsage) -> float:
        pricing = self.config.get("pricing")
        if not pricing:
            return 0.0
        return calculate_cost(usage, pricing)

    def _parse_json(self, raw: str, step: str) -> dict[str, Any]:
        """Parse JSON from LLM output, stripping markdown fences when present."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning(f"[{step}] Failed to parse JSON: {e}. Raw output: {raw[:200]}")
            return {}

    def _truncate(self, text: str) -> str:
        max_len = self.config.get("max_document_length", 2000)
        if len(text) > max_len:
            logger.warning(f"Document truncated from {len(text)} to {max_len} chars.")
        return text[:max_len]
