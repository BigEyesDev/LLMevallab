"""Single source of truth for metric display metadata — shared by dashboard and docs."""

from __future__ import annotations

from dataclasses import dataclass

from src.evaluations.evaluator import get_task_metrics
from src.pipeline.orchestrator import PipelineTask

# Dataframe columns that are operational, not quality scores.
OPERATIONAL_COLUMNS = frozenset({
    "Model",
    "Avg In Tokens",
    "Avg Out Tokens",
    "Total Cost ($)",
    "Cost/Doc ($)",
    "Avg Latency (ms)",
    "Docs",
})

# Preferred Y-axis for cost-vs-quality scatter (first match wins).
_SCATTER_PREFERENCE = ("bertscore", "comet", "llm_judge", "rouge_l", "rouge", "bleu")


@dataclass(frozen=True)
class MetricMeta:
    """Display and UX metadata for one evaluation metric."""

    key: str
    display: str
    short: str
    range: str
    good: str
    detail: str
    thresholds: tuple[tuple[float, str], ...]
    aliases: tuple[str, ...] = ()


METRIC_REGISTRY: dict[str, MetricMeta] = {
    "bleu": MetricMeta(
        key="bleu",
        display="BLEU",
        short="N-gram word overlap vs. reference. Higher = closer word-for-word match.",
        range="0 – 1",
        good="> 0.30 for translation",
        detail=(
            "Counts how many short word sequences (n-grams) appear in both the model output "
            "and the human reference. Fast to compute and widely used, but does not capture "
            "paraphrases — a perfect synonym scores zero."
        ),
        thresholds=((0.40, "excellent"), (0.30, "good"), (0.15, "moderate"), (0.0, "low")),
    ),
    "rouge_l": MetricMeta(
        key="rouge_l",
        display="ROUGE-L",
        short="Longest common subsequence vs. reference. Captures sentence structure.",
        range="0 – 1",
        good="> 0.25 for news summarisation",
        detail=(
            "Finds the longest sequence of words that appear in both texts (in order, not "
            "necessarily contiguous). Rewards outputs that preserve the flow and structure of "
            "the reference. The standard metric for news summarisation benchmarks."
        ),
        thresholds=((0.45, "excellent"), (0.30, "good"), (0.15, "moderate"), (0.0, "low")),
        aliases=("rouge",),
    ),
    "bertscore": MetricMeta(
        key="bertscore",
        display="BERTScore",
        short="Semantic similarity via BERT embeddings. Robust to paraphrasing.",
        range="0 – 1",
        good="> 0.70 (F1, DeBERTa model)",
        detail=(
            "Embeds each token with a pre-trained BERT model and computes cosine similarity "
            "between output and reference. A paraphrase that conveys the same meaning "
            "scores highly even if no words overlap. Most correlated with human judgment."
        ),
        thresholds=((0.80, "excellent"), (0.70, "good"), (0.60, "moderate"), (0.0, "low")),
    ),
    "comet": MetricMeta(
        key="comet",
        display="COMET",
        short="Neural MT metric trained on human judgments. Strong for translation quality.",
        range="0 – 1",
        good="> 0.80 for translation",
        detail=(
            "Crosslingual Optimized Metric for Evaluation of Translation. Uses source text, "
            "model output, and reference together. Generally more correlated with human "
            "translation quality than BLEU alone."
        ),
        thresholds=((0.90, "excellent"), (0.80, "good"), (0.65, "moderate"), (0.0, "low")),
    ),
    "llm_judge": MetricMeta(
        key="llm_judge",
        display="LLM Judge",
        short="Separate LLM scores faithfulness, completeness, and coherence (normalized 0–1).",
        range="0 – 1",
        good="> 0.75 for summarisation",
        detail=(
            "An independent judge model reads the source document and summary, rating "
            "faithfulness (no hallucination), completeness (covers key points), and "
            "coherence (readable flow). Scores are averaged and normalized from a 1–5 scale."
        ),
        thresholds=((0.85, "excellent"), (0.75, "good"), (0.60, "moderate"), (0.0, "low")),
    ),
}

# Build alias → canonical key lookup once.
_KEY_ALIASES: dict[str, str] = {}
for _meta in METRIC_REGISTRY.values():
    _KEY_ALIASES[_meta.key] = _meta.key
    for _alias in _meta.aliases:
        _KEY_ALIASES[_alias] = _meta.key

_DISPLAY_TO_KEY: dict[str, str] = {m.display: m.key for m in METRIC_REGISTRY.values()}


def normalize_metric_key(key: str) -> str:
    """Map config or report keys (e.g. rouge) to canonical registry key (rouge_l)."""
    return _KEY_ALIASES.get(key, key)


def get_metric_meta(key: str) -> MetricMeta | None:
    """Return metadata for a metric key or alias, or None if unknown."""
    canonical = normalize_metric_key(key)
    return METRIC_REGISTRY.get(canonical)


def display_name(key: str) -> str:
    """Human-readable label for a metric key; falls back to uppercased key."""
    meta = get_metric_meta(key)
    if meta:
        return meta.display
    return key.replace("_", " ").title()


def get_task_metric_keys(config: dict, task: str) -> list[str]:
    """Metric keys configured for a task (canonical keys from registry when known)."""
    keys = get_task_metrics(config, PipelineTask(task))
    return [normalize_metric_key(k) for k in keys]


def get_task_metric_display_names(config: dict, task: str) -> list[str]:
    """Display labels for metrics configured for a task."""
    return [display_name(k) for k in get_task_metric_keys(config, task)]


def quality_context(metric_label: str, score: float) -> str:
    """Map a metric display label or key to a human-readable verdict."""
    meta = get_metric_meta(metric_label) or METRIC_REGISTRY.get(
        _DISPLAY_TO_KEY.get(metric_label, "")
    )
    if not meta:
        return "—"
    for threshold, label in meta.thresholds:
        if score >= threshold:
            return label
    return "—"


def quality_columns_in_dataframe(df) -> list[str]:
    """Return quality metric columns present in a results dataframe."""
    return [c for c in df.columns if c not in OPERATIONAL_COLUMNS]


def metric_info_for_display(display_label: str) -> dict:
    """Return explainer dict for a display label; generic fallback for unknown metrics."""
    key = _DISPLAY_TO_KEY.get(display_label)
    meta = METRIC_REGISTRY.get(key) if key else None
    if meta:
        return {
            "short": meta.short,
            "range": meta.range,
            "good": meta.good,
            "detail": meta.detail,
        }
    return {
        "short": "Quality score for this metric.",
        "range": "0 – 1",
        "good": "Higher is better",
        "detail": "",
    }


def primary_scatter_metric(config: dict, task: str, df) -> str | None:
    """Pick the best quality column for cost-vs-quality scatter."""
    configured = get_task_metric_keys(config, task)
    display_by_key = {k: display_name(k) for k in configured}
    for pref in _SCATTER_PREFERENCE:
        canonical = normalize_metric_key(pref)
        label = display_by_key.get(canonical) or display_name(canonical)
        if label in df.columns:
            return label
    quality = quality_columns_in_dataframe(df)
    return quality[0] if quality else None
