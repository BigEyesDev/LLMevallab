from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from src.core.pricing import TokenUsage
from src.core.time import utc_now_iso

# Default per-task truncation limits (chars).  These are used when no config is
# provided — the config.yaml values take precedence at runtime.
DEFAULT_TASK_TRUNCATION_LIMITS: dict[str, int] = {
    "translation": 2000,
    "summarisation": 8000,
    "full": 4000,
}


# Ground-truth metadata key used per task — shared by orchestrator and evaluator.
TASK_GROUND_TRUTH_KEY: dict[str, str] = {
    "translation":   "reference_translation",
    "summarisation": "reference_summary",
    "full":          "reference_translation",
}


class AppModel(BaseModel):
    """Project DTO base — allows model_* field names used across the domain."""

    model_config = ConfigDict(protected_namespaces=())


class DocumentInput(AppModel):
    """Represents a raw input document before any processing."""

    doc_id: str = Field(..., description="Unique identifier for the document")
    source_language: str = Field(..., description="ISO 639-1 language code, e.g. 'de', 'fr'")
    raw_text: str = Field(..., description="Original unprocessed text")
    source: str = Field(default="unknown", description="Where the document came from")
    metadata: dict = Field(default_factory=dict, description="Any extra metadata")


class ExtractionResult(AppModel):
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


class TranslationResult(AppModel):
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


class SummaryResult(AppModel):
    """Output from the summarisation step."""

    doc_id: str
    summary: str = Field(..., description="Concise summary in English")
    key_points: list[str] = Field(default_factory=list, description="Bullet-point key takeaways")
    action_items: list[str] = Field(default_factory=list, description="Any action items or next steps")
    model_used: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    token_usage: Optional[TokenUsage] = None
    cost_usd: float = Field(default=0.0)


class TruncationInfo(AppModel):
    """Records how much of a document was actually sent to the LLM."""

    chars_original: int = Field(..., description="Character length of the raw document before truncation")
    chars_sent: int = Field(..., description="Character length actually sent to the LLM (≤ chars_original)")
    was_truncated: bool = Field(..., description="True when chars_sent < chars_original")
    limit_applied: int = Field(..., description="The per-task max_document_length limit that was in effect")


class PipelineResult(AppModel):
    """Complete result for one document, all steps combined."""

    document: DocumentInput
    extraction: Optional[ExtractionResult] = None
    translation: Optional[TranslationResult] = None
    summary: Optional[SummaryResult] = None
    total_processing_time_ms: float = Field(default=0.0)
    run_timestamp: str = Field(default_factory=utc_now_iso)
    truncation: Optional[TruncationInfo] = Field(
        default=None,
        description="Truncation metadata — populated when a per-task length limit is applied",
    )
    prompt_version: Optional[str] = Field(
        default=None,
        description="Prompt template version from configs/prompts.yaml at run time",
    )


class EvaluationScore(AppModel):
    """Scores for one metric on one document."""

    doc_id: str
    metric_name: str
    score: float
    metadata: dict = Field(default_factory=dict, description="Extra info like precision/recall breakdown")


class RunManifest(AppModel):
    """Provenance record written alongside every pipeline results file.

    Every orchestrator run writes one ``{results_stem}.manifest.json``.
    Evaluator reads this manifest to verify ground truth integrity before scoring.
    """

    run_id: str = Field(..., description="Unique run ID: {timestamp}_{6-char hex}")
    app_version: str = Field(..., description="Application version at run time")
    task: str = Field(..., description="Pipeline task name")
    model_key: str = Field(..., description="Model catalog key used for this run")
    model_id: str = Field(..., description="Resolved model_id string from catalog")
    dataset_path: str = Field(..., description="Path to the input documents file")
    dataset_hash: str = Field(..., description="SHA-256 of the input documents file")
    doc_ids: list[str] = Field(default_factory=list, description="Ordered list of processed doc_ids")
    sample_size: int = Field(..., description="Number of documents processed")
    sample_indices: list[int] = Field(default_factory=list, description="Zero-based indices of sampled docs")
    ground_truth_path: str = Field(..., description="Path to the ground truth dataset file")
    ground_truth_hash: str = Field(
        ...,
        description="SHA-256 of the ground truth texts for this run's doc_ids only",
    )
    config_hash: str = Field(
        ...,
        description="SHA-256 of task-relevant config block + model catalog entry",
    )
    config_snapshot: dict = Field(
        default_factory=dict,
        description="Pipeline + model config slice captured at run time",
    )
    results_path: str = Field(..., description="Path to the results JSON produced by this run")
    document_set_name: str = Field(
        default="",
        description="Human-readable document set name from registry (e.g. 'pearl_wish')",
    )
    selection_hash: str = Field(
        default="",
        description="12-char content hash of task + doc_ids selection",
    )
    created_at: str = Field(default_factory=utc_now_iso, description="ISO-8601 UTC timestamp")


class EvaluationReport(AppModel):
    """Full evaluation report across all documents and metrics."""

    model_used: str
    run_timestamp: str = Field(default_factory=utc_now_iso)
    scores: list[EvaluationScore] = Field(default_factory=list)
    aggregate: dict = Field(default_factory=dict, description="Averaged scores per metric")
    run_id: Optional[str] = Field(default=None, description="run_id from RunManifest if available")
    manifest_path: Optional[str] = Field(default=None, description="Path to the RunManifest used")


class ModelBenchmarkResult(AppModel):
    """Aggregated benchmark metrics for one model on one task."""

    model_key: str
    model_id: str
    quality_metrics: dict[str, float] = Field(default_factory=dict)
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0
    total_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
    n_docs: int = 0


class BenchmarkReport(AppModel):
    """Side-by-side comparison of multiple models on the same sample."""

    task: str
    run_timestamp: str = Field(default_factory=utc_now_iso)
    sample_size: int
    results: list[ModelBenchmarkResult] = Field(default_factory=list)
    prompt_version: Optional[str] = Field(
        default=None,
        description="Prompt template version used for this benchmark run",
    )
    doc_ids: list[str] = Field(
        default_factory=list,
        description="Ordered doc_ids used in this benchmark (shared across all models)",
    )
    document_set_name: str = Field(
        default="",
        description="Human-readable document set name from registry (e.g. 'pearl_wish')",
    )
    selection_hash: str = Field(
        default="",
        description="12-char content hash of task + doc_ids selection",
    )
