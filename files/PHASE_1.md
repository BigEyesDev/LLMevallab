# 📘 Phase 1 — Extraction, Translation & Summarisation Pipeline

> **Difficulty:** Beginner-friendly with senior-level architecture
> **Model:** Gemini 1.5 Pro (swappable)
> **Dataset:** EuroParl Corpus (German → English)
> **Time to complete:** ~3–5 hours (read + run)

---

## 🎯 What You Will Build in Phase 1

By the end of this phase, you will have a **fully working pipeline** that:

1. Downloads and loads real multilingual documents (German EU Parliament text)
2. Uses Gemini to extract structured data (dates, deadlines, key entities, topics)
3. Translates German documents to English
4. Summarises with structured key points
5. Runs three evaluation metrics (BLEU, ROUGE, BERTScore) against reference outputs
6. Saves all results to JSON for use in Phase 2

You will also understand **why** the architecture is designed the way it is — not just how to run it.

---

## 🧱 Architecture Overview

Before any code — understand the design. This is what makes the difference between a script and a portfolio project.

```
┌─────────────────────────────────────────────────┐
│                  PIPELINE FLOW                  │
│                                                 │
│  Raw Document                                   │
│       │                                         │
│       ▼                                         │
│  DataLoader ──► DocumentInput (Pydantic model)  │
│       │                                         │
│       ▼                                         │
│  BaseDocumentProcessor (abstract interface)     │
│       │                                         │
│       ├──► GeminiProcessor                      │
│       ├──► ClaudeProcessor   (Phase 2)          │
│       └──► OpenAIProcessor   (Phase 2)          │
│                │                                │
│                ▼                                │
│        ExtractionResult                         │
│        TranslationResult                        │
│        SummaryResult                            │
│                │                                │
│                ▼                                │
│          Evaluator                              │
│          (BLEU / ROUGE / BERTScore)             │
│                │                                │
│                ▼                                │
│         EvaluationReport → outputs/             │
└─────────────────────────────────────────────────┘
```

**Key pattern: Dependency Inversion**
The pipeline never imports `GeminiProcessor` directly. It only knows about `BaseDocumentProcessor`. This means tomorrow you can pass in `ClaudeProcessor` and nothing else changes.

---

## 📁 Phase 1 File Structure

```
multilingual-doc-intelligence/
│
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── base_processor.py       ← Abstract interface (build this first)
│   │   └── models.py               ← Pydantic data models
│   │
│   ├── providers/
│   │   ├── __init__.py
│   │   └── gemini_processor.py     ← Gemini implementation
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── data_loader.py          ← EuroParl dataset download + prep
│   │   └── orchestrator.py         ← Runs the full pipeline
│   │
│   └── evaluation/
│       ├── __init__.py
│       └── metrics.py              ← BLEU, ROUGE, BERTScore
│
├── configs/
│   ├── config.yaml
│   └── prompts.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── ground_truth/
│
├── outputs/
│   └── results/
│
├── notebooks/
│   └── 01_extraction_translation_summarisation.ipynb
│
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## 🔧 Step 0 — Environment Setup

### `requirements.txt`

```txt
# Core
python-dotenv==1.0.1
pydantic==2.7.1
pyyaml==6.0.1
requests==2.32.3

# LLM Providers
google-generativeai==0.7.2

# Evaluation Metrics
nltk==3.8.1
rouge-score==0.1.2
bert-score==0.3.13
torch==2.3.1
transformers==4.41.2

# Data & Utils
datasets==2.19.1          # HuggingFace datasets
pandas==2.2.2
numpy==1.26.4
tqdm==4.66.4

# Notebook
jupyter==1.0.0
ipykernel==6.29.4

# Testing
pytest==8.2.0
```

### `.env.example`

```env
# Copy this file to .env and fill in your keys
# NEVER commit .env to git

GEMINI_API_KEY=your_gemini_api_key_here

# Phase 2 (leave blank for now)
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
```

### `.gitignore`

```gitignore
# Environment
.env
venv/
__pycache__/
*.pyc

# Data (too large for git)
data/raw/
data/processed/

# Outputs
outputs/results/

# Jupyter checkpoints
.ipynb_checkpoints/

# OS
.DS_Store
```

---

## 📐 Step 1 — Data Models (`src/core/models.py`)

> 💡 **Why Pydantic?**
> Pydantic gives you data validation for free. If the LLM returns garbage, you catch it here — not silently downstream. It also gives you free JSON serialisation.

```python
# src/core/models.py

from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class DocumentInput(BaseModel):
    """Represents a raw input document before any processing."""

    doc_id: str = Field(..., description="Unique identifier for the document")
    source_language: str = Field(..., description="ISO 639-1 language code, e.g. 'de', 'fr'")
    raw_text: str = Field(..., description="Original unprocessed text")
    source: str = Field(default="unknown", description="Where the document came from")
    metadata: dict = Field(default_factory=dict, description="Any extra metadata")


class ExtractionResult(BaseModel):
    """Structured output from the extraction step."""

    doc_id: str
    entities: list[str] = Field(default_factory=list, description="Named entities: people, orgs, places")
    dates: list[str] = Field(default_factory=list, description="All dates found in the document")
    deadlines: list[str] = Field(default_factory=list, description="Specific deadline mentions")
    topics: list[str] = Field(default_factory=list, description="Main topics or themes")
    key_clauses: list[str] = Field(default_factory=list, description="Important clauses or statements")
    raw_llm_output: str = Field(default="", description="Original LLM response before parsing")
    model_used: str = Field(default="", description="Which model produced this")
    processing_time_ms: float = Field(default=0.0)


class TranslationResult(BaseModel):
    """Output from the translation step."""

    doc_id: str
    source_language: str
    target_language: str = "en"
    original_text: str
    translated_text: str
    model_used: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)


class SummaryResult(BaseModel):
    """Output from the summarisation step."""

    doc_id: str
    summary: str = Field(..., description="Concise summary in English")
    key_points: list[str] = Field(default_factory=list, description="Bullet-point key takeaways")
    action_items: list[str] = Field(default_factory=list, description="Any action items or next steps")
    model_used: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)


class PipelineResult(BaseModel):
    """Complete result for one document, all steps combined."""

    document: DocumentInput
    extraction: Optional[ExtractionResult] = None
    translation: Optional[TranslationResult] = None
    summary: Optional[SummaryResult] = None
    total_processing_time_ms: float = Field(default=0.0)
    run_timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class EvaluationScore(BaseModel):
    """Scores for one metric on one document."""

    doc_id: str
    metric_name: str
    score: float
    metadata: dict = Field(default_factory=dict, description="Extra info like precision/recall breakdown")


class EvaluationReport(BaseModel):
    """Full evaluation report across all documents and metrics."""

    model_used: str
    run_timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    scores: list[EvaluationScore] = Field(default_factory=list)
    aggregate: dict = Field(default_factory=dict, description="Averaged scores per metric")
```

---

## 🧩 Step 2 — Abstract Base Class (`src/core/base_processor.py`)

> 💡 **Why an abstract base class?**
> This is the heart of the scalable design. Every model — Gemini, Claude, GPT-4 — must implement the same three methods. Your pipeline calls `processor.translate(doc)` and doesn't care *which* processor it is. This is the **Open/Closed Principle** in practice.

```python
# src/core/base_processor.py

from abc import ABC, abstractmethod
from src.core.models import DocumentInput, ExtractionResult, TranslationResult, SummaryResult


class BaseDocumentProcessor(ABC):
    """
    Abstract base class for all LLM document processors.

    All providers (Gemini, Claude, OpenAI, open-source) must inherit
    from this class and implement all three abstract methods.

    This ensures the pipeline is fully model-agnostic — swapping models
    requires zero changes to orchestration code.

    Usage:
        class GeminiProcessor(BaseDocumentProcessor):
            def extract(self, document: DocumentInput) -> ExtractionResult:
                ...
    """

    def __init__(self, model_name: str, config: dict):
        """
        Args:
            model_name: Human-readable model identifier (e.g. 'gemini-1.5-pro')
            config: Model-specific configuration dict from config.yaml
        """
        self.model_name = model_name
        self.config = config

    @abstractmethod
    def extract(self, document: DocumentInput) -> ExtractionResult:
        """
        Extract structured information from the document.

        Must return an ExtractionResult with entities, dates,
        deadlines, topics, and key clauses populated.
        """
        ...

    @abstractmethod
    def translate(self, document: DocumentInput, target_language: str = "en") -> TranslationResult:
        """
        Translate the document text to target_language.

        If the document is already in target_language, return
        a TranslationResult with translated_text == original_text.
        """
        ...

    @abstractmethod
    def summarise(self, document: DocumentInput) -> SummaryResult:
        """
        Produce a structured summary with key points and action items.

        Summary should always be in English regardless of source language.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name})"
```

---

## 🔑 Step 3 — Centralised Configuration

### `configs/config.yaml`

```yaml
# configs/config.yaml
# All tunable parameters live here — never hardcode in source files

pipeline:
  target_language: "en"
  max_document_length: 2000       # chars — truncate longer docs for cost control
  batch_size: 5                   # documents per run
  save_raw_llm_output: true       # useful for debugging

models:
  default: "gemini"

  gemini:
    model_id: "gemini-1.5-pro"
    temperature: 0.1              # low temp = more deterministic, better for extraction
    max_output_tokens: 2048
    timeout_seconds: 30

  claude:                         # Phase 2
    model_id: "claude-sonnet-4-6"
    temperature: 0.1
    max_output_tokens: 2048

  openai:                         # Phase 2
    model_id: "gpt-4o"
    temperature: 0.1
    max_output_tokens: 2048

dataset:
  name: "europarl"
  language_pairs:
    - ["de", "en"]               # German → English (Phase 1)
  sample_size: 20                # how many docs to use for Phase 1
  ground_truth_path: "data/ground_truth/europarl_references.json"

evaluation:
  metrics:
    - bleu
    - rouge
    - bertscore
  bertscore_model: "microsoft/deberta-xlarge-mnli"
  output_path: "outputs/results/"

paths:
  raw_data: "data/raw/"
  processed_data: "data/processed/"
  outputs: "outputs/results/"
```

### `configs/prompts.yaml`

> 💡 **Why externalise prompts?**
> Prompts change constantly during development. If they're in code, every tweak is a code change. In a YAML file, you can version them, A/B test them, and let non-engineers edit them. This is a real production pattern.

```yaml
# configs/prompts.yaml
# Centralised prompt templates — {text} is replaced at runtime

extraction:
  system: |
    You are a precise document analysis assistant. Your job is to extract
    structured information from formal documents. Always respond with valid
    JSON only — no markdown, no explanation, no extra text.

  user: |
    Analyse the following document and extract structured information.

    Document text:
    {text}

    Respond with ONLY this JSON structure (no markdown, no extra text):
    {{
      "entities": ["list of named entities: people, organisations, places"],
      "dates": ["list of all dates mentioned"],
      "deadlines": ["list of deadline-related statements"],
      "topics": ["list of 3-5 main topics"],
      "key_clauses": ["list of 3-5 important statements or clauses"]
    }}

translation:
  system: |
    You are a professional translator specialising in formal and legal documents.
    Translate accurately and preserve the original meaning and tone.
    Respond with ONLY the translated text — no explanation, no preamble.

  user: |
    Translate the following {source_language} text to English.
    Preserve formal tone and technical terminology.

    Text to translate:
    {text}

summarisation:
  system: |
    You are an expert document summariser. Create concise, structured summaries
    that capture the essential information. Always respond in English.
    Respond with ONLY valid JSON — no markdown, no extra text.

  user: |
    Summarise the following document.

    Document:
    {text}

    Respond with ONLY this JSON structure:
    {{
      "summary": "2-3 sentence overview of the document",
      "key_points": ["point 1", "point 2", "point 3"],
      "action_items": ["action 1 if any, else empty list"]
    }}
```

---

## 📥 Step 4 — Data Loader (`src/pipeline/data_loader.py`)

```python
# src/pipeline/data_loader.py

import json
import logging
from pathlib import Path
from typing import Generator

from datasets import load_dataset
from tqdm import tqdm

from src.core.models import DocumentInput

logger = logging.getLogger(__name__)


class EuroParlDataLoader:
    """
    Downloads and prepares EuroParl parallel corpus documents.

    The EuroParl corpus is EU Parliament proceedings in 21 languages.
    We use the German-English pair (de-en) as our primary language pair.

    Dataset card: https://huggingface.co/datasets/Helsinki-NLP/europarl
    """

    DATASET_NAME = "Helsinki-NLP/europarl"
    DEFAULT_LANGUAGE_PAIR = "de-en"

    def __init__(self, processed_dir: str = "data/processed/", sample_size: int = 20):
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.sample_size = sample_size

    def download_and_prepare(self, language_pair: str = DEFAULT_LANGUAGE_PAIR) -> Path:
        """
        Downloads EuroParl dataset and saves a processed subset to disk.

        Args:
            language_pair: Language pair string, e.g. 'de-en'

        Returns:
            Path to the saved processed JSON file
        """
        logger.info(f"Loading EuroParl dataset ({language_pair})...")

        dataset = load_dataset(
            self.DATASET_NAME,
            language_pair,
            split="train",
            streaming=True,   # streaming=True avoids downloading the full dataset
            trust_remote_code=True,
        )

        documents: list[dict] = []
        source_lang = language_pair.split("-")[0]  # "de"

        for i, example in enumerate(tqdm(dataset, total=self.sample_size, desc="Loading documents")):
            if i >= self.sample_size:
                break

            # EuroParl stores parallel sentences — we group them into document-length chunks
            source_text = example["translation"][source_lang]
            reference_english = example["translation"]["en"]

            doc = DocumentInput(
                doc_id=f"europarl_{language_pair}_{i:04d}",
                source_language=source_lang,
                raw_text=source_text,
                source="europarl",
                metadata={"reference_translation": reference_english},
            )
            documents.append(doc.model_dump())

        output_path = self.processed_dir / f"europarl_{language_pair}_{self.sample_size}docs.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(documents)} documents to {output_path}")
        return output_path

    def load_from_disk(self, file_path: str) -> list[DocumentInput]:
        """
        Loads preprocessed documents from disk.

        Args:
            file_path: Path to the processed JSON file

        Returns:
            List of DocumentInput objects
        """
        with open(file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        return [DocumentInput(**doc) for doc in raw]

    def load_ground_truth(self, file_path: str) -> dict[str, str]:
        """
        Loads human reference translations for evaluation.

        Returns:
            Dict mapping doc_id → reference English translation
        """
        documents = self.load_from_disk(file_path)
        return {
            doc.doc_id: doc.metadata.get("reference_translation", "")
            for doc in documents
        }
```

---

## ⚡ Step 5 — Gemini Provider (`src/providers/gemini_processor.py`)

> 💡 **Notice:** This class knows nothing about data loading, evaluation, or orchestration. It does one thing: talk to Gemini and return typed results.

```python
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
```

---

## 🎛️ Step 6 — Pipeline Orchestrator (`src/pipeline/orchestrator.py`)

> 💡 **Notice:** The orchestrator accepts `BaseDocumentProcessor` — it never imports `GeminiProcessor`. This is the dependency inversion in action.

```python
# src/pipeline/orchestrator.py

import json
import logging
import os
from pathlib import Path
from datetime import datetime

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from src.core.base_processor import BaseDocumentProcessor
from src.core.models import DocumentInput, PipelineResult
from src.providers.gemini_processor import GeminiProcessor

load_dotenv()
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_prompts(prompts_path: str = "configs/prompts.yaml") -> dict:
    with open(prompts_path, "r") as f:
        return yaml.safe_load(f)


def build_processor(model_key: str, config: dict, prompts: dict) -> BaseDocumentProcessor:
    """
    Factory function — returns the right processor for the given model key.

    Adding a new model: add one elif block here.
    Nothing else in the codebase changes.

    Args:
        model_key: Key from config.yaml, e.g. 'gemini', 'claude', 'openai'
        config: Full config dict
        prompts: Full prompts dict

    Returns:
        Configured processor instance
    """
    model_config = config["models"][model_key]

    if model_key == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set in environment.")
        return GeminiProcessor(api_key=api_key, config=model_config, prompts=prompts)

    elif model_key == "claude":
        # Phase 2
        from src.providers.claude_processor import ClaudeProcessor
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        return ClaudeProcessor(api_key=api_key, config=model_config, prompts=prompts)

    elif model_key == "openai":
        # Phase 2
        from src.providers.openai_processor import OpenAIProcessor
        api_key = os.environ.get("OPENAI_API_KEY")
        return OpenAIProcessor(api_key=api_key, config=model_config, prompts=prompts)

    else:
        raise ValueError(f"Unknown model key: '{model_key}'. Add it to build_processor().")


class PipelineOrchestrator:
    """
    Runs the full document processing pipeline.

    Steps per document:
        1. Extract structured data
        2. Translate to English
        3. Summarise

    Results are saved as JSON for downstream evaluation.
    """

    def __init__(self, processor: BaseDocumentProcessor, config: dict):
        self.processor = processor
        self.config = config
        self.output_dir = Path(config["paths"]["outputs"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, documents: list[DocumentInput]) -> list[PipelineResult]:
        """
        Processes a list of documents through the full pipeline.

        Args:
            documents: List of DocumentInput objects

        Returns:
            List of PipelineResult objects with extraction, translation, summary
        """
        results: list[PipelineResult] = []
        target_lang = self.config["pipeline"]["target_language"]

        logger.info(f"Starting pipeline with {self.processor} on {len(documents)} documents.")

        for doc in tqdm(documents, desc=f"Processing [{self.processor.model_name}]"):
            result = self._process_single(doc, target_lang)
            results.append(result)

        self._save_results(results)
        return results

    def _process_single(self, document: DocumentInput, target_language: str) -> PipelineResult:
        """Runs all three steps on one document, capturing errors gracefully."""
        import time
        start = time.time()

        extraction = None
        translation = None
        summary = None

        try:
            extraction = self.processor.extract(document)
        except Exception as e:
            logger.error(f"[{document.doc_id}] Extraction failed: {e}")

        try:
            translation = self.processor.translate(document, target_language=target_language)
        except Exception as e:
            logger.error(f"[{document.doc_id}] Translation failed: {e}")

        try:
            # Summarise from translated text if available, else raw
            summary_input = document
            if translation:
                summary_input = document.model_copy(
                    update={"raw_text": translation.translated_text, "source_language": target_language}
                )
            summary = self.processor.summarise(summary_input)
        except Exception as e:
            logger.error(f"[{document.doc_id}] Summarisation failed: {e}")

        return PipelineResult(
            document=document,
            extraction=extraction,
            translation=translation,
            summary=summary,
            total_processing_time_ms=(time.time() - start) * 1000,
        )

    def _save_results(self, results: list[PipelineResult]) -> Path:
        """Saves results to a timestamped JSON file."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        model_name = self.processor.model_name.replace("/", "_").replace("-", "_")
        filename = f"results_{model_name}_{timestamp}.json"
        output_path = self.output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                [r.model_dump() for r in results],
                f,
                ensure_ascii=False,
                indent=2,
                default=str,
            )

        logger.info(f"Results saved to {output_path}")
        return output_path


# ─────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Run the document processing pipeline.")
    parser.add_argument("--model", default="gemini", help="Model key (gemini, claude, openai)")
    parser.add_argument("--input", required=True, help="Path to processed documents JSON")
    parser.add_argument("--sample", type=int, default=5, help="Number of documents to process")
    args = parser.parse_args()

    config = load_config()
    prompts = load_prompts()

    processor = build_processor(args.model, config, prompts)

    from src.pipeline.data_loader import EuroParlDataLoader
    loader = EuroParlDataLoader()
    documents = loader.load_from_disk(args.input)[: args.sample]

    orchestrator = PipelineOrchestrator(processor=processor, config=config)
    results = orchestrator.run(documents)

    print(f"\n✅ Pipeline complete. Processed {len(results)} documents.")
    print(f"   Results saved to: {config['paths']['outputs']}")
```

---

## 📊 Step 7 — Evaluation Metrics (`src/evaluation/metrics.py`)

> 💡 **Before reading the code — understand the metrics:**
>
> | Metric | Intuition | Range | Good Score |
> |---|---|---|---|
> | **BLEU** | "Did the LLM use the same words as the reference?" Counts matching n-grams (1,2,3,4-word sequences) | 0–1 | > 0.4 for translation |
> | **ROUGE-L** | "Did the LLM capture the reference ideas?" Measures longest matching subsequence | 0–1 | > 0.5 for summarisation |
> | **BERTScore** | "Does the LLM output *mean* the same thing?" Uses deep embeddings to compare meaning, not just words | 0–1 | > 0.85 |
>
> **When BLEU fails:** BLEU rewards exact word matches. "The car is red" vs "The automobile is crimson" scores near 0, even though they mean the same thing. This is why BERTScore exists.
>
> **When BERTScore fails:** It can miss factual errors if the text sounds semantically similar but has wrong numbers or names. Always use multiple metrics.

```python
# src/evaluation/metrics.py

import logging
from dataclasses import dataclass

import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn

from src.core.models import EvaluationScore

logger = logging.getLogger(__name__)

# Download NLTK tokeniser data if not present
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


@dataclass
class MetricInput:
    """Holds hypothesis and reference for one document."""
    doc_id: str
    hypothesis: str    # LLM output (what we're evaluating)
    reference: str     # Ground truth (what it should be)


class BLEUMetric:
    """
    BLEU (Bilingual Evaluation Understudy) Score.

    Originally designed for machine translation evaluation (Papineni et al., 2002).
    Measures how much of the LLM's output appears in the reference, using
    n-gram overlap (1-gram to 4-gram).

    Score range: 0.0 (no overlap) to 1.0 (perfect match)
    Typical use: translation quality

    Paper: https://aclanthology.org/P02-1040/
    """

    name = "bleu"

    def score(self, inputs: list[MetricInput]) -> list[EvaluationScore]:
        """
        Computes sentence-level BLEU for each input, plus corpus-level BLEU.

        Args:
            inputs: List of hypothesis/reference pairs

        Returns:
            List of EvaluationScore — one per document, plus one corpus-level score
        """
        scores: list[EvaluationScore] = []
        smoothing = SmoothingFunction().method1  # avoids zero for short texts

        for item in inputs:
            hypothesis_tokens = item.hypothesis.lower().split()
            reference_tokens = [item.reference.lower().split()]  # BLEU expects list of refs

            sentence_score = sentence_bleu(
                references=reference_tokens,
                hypothesis=hypothesis_tokens,
                smoothing_function=smoothing,
            )

            scores.append(EvaluationScore(
                doc_id=item.doc_id,
                metric_name=self.name,
                score=round(sentence_score, 4),
            ))

        # Corpus-level BLEU (more stable than sentence average)
        all_refs = [[item.reference.lower().split()] for item in inputs]
        all_hyps = [item.hypothesis.lower().split() for item in inputs]
        corpus_score = corpus_bleu(all_refs, all_hyps, smoothing_function=smoothing)

        scores.append(EvaluationScore(
            doc_id="__corpus__",
            metric_name=f"{self.name}_corpus",
            score=round(corpus_score, 4),
        ))

        logger.info(f"BLEU corpus score: {corpus_score:.4f}")
        return scores


class ROUGEMetric:
    """
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation).

    Designed for summarisation evaluation (Lin, 2004).
    We use ROUGE-L which measures the longest common subsequence —
    it captures sentence-level structure, not just n-gram matches.

    Score range: 0.0 to 1.0
    Typical use: summarisation quality

    Paper: https://aclanthology.org/W04-1013/
    """

    name = "rouge_l"

    def __init__(self):
        self._scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def score(self, inputs: list[MetricInput]) -> list[EvaluationScore]:
        scores: list[EvaluationScore] = []

        for item in inputs:
            result = self._scorer.score(
                target=item.reference,
                prediction=item.hypothesis,
            )
            rouge_l = result["rougeL"]

            scores.append(EvaluationScore(
                doc_id=item.doc_id,
                metric_name=self.name,
                score=round(rouge_l.fmeasure, 4),
                metadata={
                    "precision": round(rouge_l.precision, 4),
                    "recall": round(rouge_l.recall, 4),
                    "f1": round(rouge_l.fmeasure, 4),
                },
            ))

        avg = sum(s.score for s in scores) / len(scores) if scores else 0
        logger.info(f"ROUGE-L average F1: {avg:.4f}")
        return scores


class BERTScoreMetric:
    """
    BERTScore — Semantic Similarity via Contextual Embeddings.

    Instead of counting matching words, BERTScore encodes both hypothesis
    and reference using a pretrained transformer (DeBERTa), then computes
    cosine similarity between token embeddings.

    This means semantically equivalent sentences score high even if they
    use completely different words — a key advantage over BLEU/ROUGE.

    Score range: typically 0.80–1.0 for English (due to embedding similarity floor)
    Typical use: translation AND summarisation

    Paper: https://arxiv.org/abs/1904.09675
    """

    name = "bertscore"

    def __init__(self, model_type: str = "microsoft/deberta-xlarge-mnli", device: str = "cpu"):
        self.model_type = model_type
        self.device = device

    def score(self, inputs: list[MetricInput]) -> list[EvaluationScore]:
        hypotheses = [item.hypothesis for item in inputs]
        references = [item.reference for item in inputs]

        logger.info(f"Computing BERTScore with {self.model_type} on {len(inputs)} docs...")

        # bert_score returns tensors of P, R, F1 per sentence
        precisions, recalls, f1s = bert_score_fn(
            cands=hypotheses,
            refs=references,
            model_type=self.model_type,
            device=self.device,
            verbose=False,
        )

        scores: list[EvaluationScore] = []
        for i, item in enumerate(inputs):
            scores.append(EvaluationScore(
                doc_id=item.doc_id,
                metric_name=self.name,
                score=round(f1s[i].item(), 4),
                metadata={
                    "precision": round(precisions[i].item(), 4),
                    "recall": round(recalls[i].item(), 4),
                },
            ))

        avg = sum(s.score for s in scores) / len(scores) if scores else 0
        logger.info(f"BERTScore average F1: {avg:.4f}")
        return scores


class MetricsRunner:
    """
    Convenience class that runs all configured metrics in one call.

    Usage:
        runner = MetricsRunner(metrics=["bleu", "rouge", "bertscore"])
        all_scores = runner.run_all(inputs)
    """

    AVAILABLE_METRICS = {
        "bleu": BLEUMetric,
        "rouge": ROUGEMetric,
        "bertscore": BERTScoreMetric,
    }

    def __init__(self, metrics: list[str], bertscore_model: str = "microsoft/deberta-xlarge-mnli"):
        self._metrics = []
        for name in metrics:
            if name not in self.AVAILABLE_METRICS:
                raise ValueError(f"Unknown metric: '{name}'. Available: {list(self.AVAILABLE_METRICS)}")
            if name == "bertscore":
                self._metrics.append(BERTScoreMetric(model_type=bertscore_model))
            else:
                self._metrics.append(self.AVAILABLE_METRICS[name]())

    def run_all(self, inputs: list[MetricInput]) -> dict[str, list[EvaluationScore]]:
        """
        Runs all configured metrics.

        Returns:
            Dict mapping metric_name → list of EvaluationScore
        """
        return {
            metric.name: metric.score(inputs)
            for metric in self._metrics
        }
```

---

## 🚀 Step 8 — Quick Start Walkthrough

Follow these commands in order to run Phase 1 end-to-end:

```bash
# ── 1. Setup ──────────────────────────────────────────────
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# → Add your GEMINI_API_KEY to .env

# ── 2. Download dataset ──────────────────────────────────
python -c "
from src.pipeline.data_loader import EuroParlDataLoader
loader = EuroParlDataLoader(sample_size=20)
path = loader.download_and_prepare('de-en')
print(f'Dataset saved to: {path}')
"

# ── 3. Run the pipeline (5 documents to start) ──────────
python src/pipeline/orchestrator.py \
  --model gemini \
  --input data/processed/europarl_de-en_20docs.json \
  --sample 5

# ── 4. Check your outputs ────────────────────────────────
ls outputs/results/
cat outputs/results/results_gemini_*.json | python -m json.tool | head -80
```

---

## 📓 Notebook 01 Outline (`notebooks/01_extraction_translation_summarisation.ipynb`)

> The notebook runs the same code interactively, with explanations at every cell.

```
Cell 1  — Introduction & Learning Goals (Markdown)
Cell 2  — Install dependencies (pip install)
Cell 3  — Explain: What is EuroParl? Why use it? (Markdown)
Cell 4  — Load and preview the dataset (DataFrame view)
Cell 5  — Explain: What is a base class and why? (Markdown)
Cell 6  — Instantiate GeminiProcessor (show the config)
Cell 7  — Run extraction on ONE document — print result
Cell 8  — Run translation on ONE document — print result
Cell 9  — Run summarisation on ONE document — print result
Cell 10 — Run full pipeline on 5 documents
Cell 11 — Explain: What is BLEU? (Markdown + diagram)
Cell 12 — Compute BLEU on translation results
Cell 13 — Explain: What is ROUGE? (Markdown)
Cell 14 — Compute ROUGE on summaries
Cell 15 — Explain: What is BERTScore? (Markdown)
Cell 16 — Compute BERTScore
Cell 17 — Visualise: bar chart of all three metrics
Cell 18 — Analysis: Where did the model do well? Where did it fail? (Markdown + your notes)
Cell 19 — Save results for Phase 2
Cell 20 — Summary & What's Next (Markdown)
```

---

## ✅ Phase 1 Completion Checklist

Before moving to Phase 2, verify:

- [ ] `src/core/models.py` — all Pydantic models defined
- [ ] `src/core/base_processor.py` — abstract class with three abstract methods
- [ ] `src/providers/gemini_processor.py` — full implementation, all three methods working
- [ ] `src/pipeline/data_loader.py` — downloads and saves EuroParl subset
- [ ] `src/pipeline/orchestrator.py` — runs pipeline, saves JSON output
- [ ] `src/evaluation/metrics.py` — BLEU, ROUGE, BERTScore all producing scores
- [ ] `configs/config.yaml` — all parameters externalised
- [ ] `configs/prompts.yaml` — all prompts externalised
- [ ] Notebook 01 runs end-to-end without errors
- [ ] At least 10 documents processed and scored
- [ ] Results saved to `outputs/results/`

---

## 🔮 What Comes in Phase 2

- Add Claude, OpenAI, and an open-source model (same three-method interface)
- Add COMET — the current state-of-the-art MT metric
- Add LLM-as-Judge — use Claude to evaluate Gemini's output
- Build `src/evaluation/benchmark.py` to run all four models side-by-side
- Head-to-head comparison tables and visualisations

---

*Phase 1 — Started May 2026. Built from first principles.*
