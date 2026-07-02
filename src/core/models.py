from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

from src.core.pricing import TokenUsage


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
    token_usage: Optional[TokenUsage] = None
    cost_usd: float = Field(default=0.0)


class TranslationResult(BaseModel):
    """Output from the translation step."""

    doc_id: str
    source_language: str
    target_language: str = "en"
    original_text: str
    translated_text: str
    model_used: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    token_usage: Optional[TokenUsage] = None
    cost_usd: float = Field(default=0.0)


class SummaryResult(BaseModel):
    """Output from the summarisation step."""

    doc_id: str
    summary: str = Field(..., description="Concise summary in English")
    key_points: list[str] = Field(default_factory=list, description="Bullet-point key takeaways")
    action_items: list[str] = Field(default_factory=list, description="Any action items or next steps")
    model_used: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    token_usage: Optional[TokenUsage] = None
    cost_usd: float = Field(default=0.0)


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


class ModelBenchmarkResult(BaseModel):
    """Aggregated benchmark metrics for one model on one task."""

    model_key: str
    model_id: str
    quality_metrics: dict[str, float] = Field(default_factory=dict)
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0
    total_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
    n_docs: int = 0


class BenchmarkReport(BaseModel):
    """Side-by-side comparison of multiple models on the same sample."""

    task: str
    run_timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    sample_size: int
    results: list[ModelBenchmarkResult] = Field(default_factory=list)