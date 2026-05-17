# src/providers/gemini_processor.py

import json
import logging
import time
from typing import Any

import google.generativeai as genai

from src.core.base_processor import BaseDocumentProcessor
from src.core.models import DocumentInput, ExtractionResult, TranslationResult, SummaryResult

logger = logging.getLogger(__name__)


class GeminiProcessor(BaseDocumentProcessor):
    """
    Gemini implementation of BaseDocumentProcessor.

    Uses Gemini 1.5 Pro for extraction, translation, and summarisation.
    All prompts are loaded from configs/prompts.yaml — not hardcoded here.

    API reference: https://ai.google.dev/api/python/google/generativeai
    """

    def __init__(self, api_key: str, config: dict, prompts: dict):
        """
        Args:
            api_key: Gemini API key
            config: Model config block from config.yaml (temperature, tokens, etc.)
            prompts: Prompt templates from prompts.yaml
        """
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

    # ─────────────────────────────────────────────
    # Public interface (implements BaseDocumentProcessor)
    # ─────────────────────────────────────────────

    def extract(self, document: DocumentInput) -> ExtractionResult:
        """Extract entities, dates, deadlines, topics, and key clauses."""
        start = time.time()

        prompt = self._build_prompt(
            step="extraction",
            variables={"text": self._truncate(document.raw_text)},
        )
        raw_output = self._call_api(prompt)
        parsed = self._parse_json(raw_output, step="extraction")

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
        )

    def translate(self, document: DocumentInput, target_language: str = "en") -> TranslationResult:
        """Translate document text to target language."""
        start = time.time()

        # Skip translation if already in target language
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
        translated_text = self._call_api(prompt)

        return TranslationResult(
            doc_id=document.doc_id,
            source_language=document.source_language,
            target_language=target_language,
            original_text=document.raw_text,
            translated_text=translated_text.strip(),
            model_used=self.model_name,
            processing_time_ms=(time.time() - start) * 1000,
        )

    def summarise(self, document: DocumentInput) -> SummaryResult:
        """Produce a structured English summary."""
        start = time.time()

        prompt = self._build_prompt(
            step="summarisation",
            variables={"text": self._truncate(document.raw_text)},
        )
        raw_output = self._call_api(prompt)
        parsed = self._parse_json(raw_output, step="summarisation")

        return SummaryResult(
            doc_id=document.doc_id,
            summary=parsed.get("summary", ""),
            key_points=parsed.get("key_points", []),
            action_items=parsed.get("action_items", []),
            model_used=self.model_name,
            processing_time_ms=(time.time() - start) * 1000,
        )

    # ─────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────

    def _build_prompt(self, step: str, variables: dict[str, str]) -> str:
        """
        Builds a full prompt by injecting variables into the template.

        Args:
            step: One of 'extraction', 'translation', 'summarisation'
            variables: Dict of {placeholder: value} to inject

        Returns:
            Combined system + user prompt string
        """
        step_prompts = self._prompts[step]
        system = step_prompts["system"]
        user = step_prompts["user"].format(**variables)
        return f"{system}\n\n{user}"

    def _call_api(self, prompt: str) -> str:
        """
        Makes a single API call to Gemini.

        Raises:
            RuntimeError: If the API call fails after retries
        """
        try:
            response = self._client.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise RuntimeError(f"Gemini API error: {e}") from e

    def _parse_json(self, raw: str, step: str) -> dict[str, Any]:
        """
        Safely parses JSON from LLM output.

        LLMs sometimes wrap JSON in markdown fences — this handles that.
        Returns empty dict on failure rather than crashing the pipeline.
        """
        # Strip markdown code fences if present
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
        """Truncate text to max_document_length to control costs."""
        max_len = self.config.get("max_document_length", 2000)
        if len(text) > max_len:
            logger.warning(f"Document truncated from {len(text)} to {max_len} chars.")
        return text[:max_len]