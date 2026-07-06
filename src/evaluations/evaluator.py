# src/evaluation/evaluator.py

import json
import logging
import statistics
import warnings
from pathlib import Path

import yaml

from src.core.config import get_processed_path
from src.core.time import utc_timestamp
from src.core.models import (
    EvaluationReport,
    EvaluationScore,
    PipelineResult,
    RunManifest,
    TASK_GROUND_TRUTH_KEY,
)
from src.core.provenance import ground_truth_hash as compute_ground_truth_hash
from src.core.config import get_processed_path
from src.evaluations.metrics import MetricInput, MetricsRunner
from src.pipeline.orchestrator import PipelineTask

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
# Task → Metric mapping
# ─────────────────────────────────────────────────────

# Which metrics apply to which task.
# This prevents running BLEU on summaries or ROUGE on translations —
# each metric is only meaningful in the right context.
TASK_METRICS: dict[PipelineTask, list[str]] = {
    PipelineTask.TRANSLATION:   ["bleu", "bertscore", "comet"],
    PipelineTask.SUMMARISATION: ["rouge", "bertscore"],
    PipelineTask.FULL:          ["bleu", "rouge", "bertscore", "comet"],
}


def get_task_metrics(config: dict, task: PipelineTask) -> list[str]:
    """Return metrics for *task* from config, falling back to TASK_METRICS."""
    metrics_cfg = config.get("evaluation", {}).get("metrics", {})
    if task.value in metrics_cfg:
        return metrics_cfg[task.value]
    return TASK_METRICS[task]

# Which field in the pipeline result to use as the hypothesis per task
TASK_HYPOTHESIS_FIELD: dict[PipelineTask, str] = {
    PipelineTask.TRANSLATION:   "translated_text",   # from TranslationResult
    PipelineTask.SUMMARISATION: "summary",           # from SummaryResult
    PipelineTask.FULL:          "translated_text",   # default — override as needed
}

# Which metadata key in DocumentInput holds the ground truth per task.
# Kept for backward-compat with code that imports from evaluator.
# Canonical definition is now in src.core.models.TASK_GROUND_TRUTH_KEY.
TASK_GROUND_TRUTH_KEY_ENUM: dict[PipelineTask, str] = {
    PipelineTask.TRANSLATION:   "reference_translation",
    PipelineTask.SUMMARISATION: "reference_summary",
    PipelineTask.FULL:          "reference_translation",
}


# ─────────────────────────────────────────────────────
# Result & Ground Truth Loaders
# ─────────────────────────────────────────────────────

def load_pipeline_results(results_path: str) -> list[dict]:
    """
    Loads pipeline output JSON produced by PipelineOrchestrator._save_results().

    Returns:
        List of raw result dicts (one per document)
    """
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    logger.info(f"Loaded {len(results)} pipeline results from {results_path}")
    return results


def load_manifest(manifest_path: str | Path) -> RunManifest:
    """Load a RunManifest from its JSON file."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return RunManifest.model_validate(data)


def load_ground_truth(ground_truth_path: str, task: PipelineTask) -> dict[str, str]:
    """
    Loads ground truth references from a processed dataset JSON.

    Reads the correct metadata key depending on the task:
        TRANSLATION   → metadata.reference_translation  (EuroParl)
        SUMMARISATION → metadata.reference_summary       (CNN/DailyMail)

    Args:
        ground_truth_path: Path to processed dataset JSON (e.g. europarl_20docs.json)
        task: Pipeline task — determines which metadata key to read

    Returns:
        Dict mapping doc_id → reference text
    """
    gt_key = TASK_GROUND_TRUTH_KEY_ENUM[task]

    with open(ground_truth_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    ground_truth: dict[str, str] = {}
    missing = 0

    for doc in documents:
        doc_id = doc["doc_id"]
        reference = doc.get("metadata", {}).get(gt_key, "")
        if not reference:
            logger.warning(f"[{doc_id}] Missing ground truth key '{gt_key}' — skipping.")
            missing += 1
            continue
        ground_truth[doc_id] = reference

    logger.info(
        f"Ground truth loaded: {len(ground_truth)} docs "
        f"(key='{gt_key}', missing={missing})"
    )
    return ground_truth


# ─────────────────────────────────────────────────────
# Manifest verification
# ─────────────────────────────────────────────────────

def verify_manifest_ground_truth(manifest: RunManifest) -> None:
    """
    Re-computes the ground truth hash for the manifest's doc_ids and compares it
    to the stored ``ground_truth_hash``.

    Raises:
        RuntimeError: if the hash does not match, indicating the ground truth file
                      has changed or is not the file used during inference.
        ValueError:   if the manifest lacks enough data to verify (empty hashes).
    """
    if not manifest.ground_truth_hash:
        raise ValueError(
            f"RunManifest '{manifest.run_id}' has no ground_truth_hash recorded. "
            "Re-run the pipeline with --dataset to enable hash verification."
        )
    if not manifest.ground_truth_path or not manifest.doc_ids:
        raise ValueError(
            f"RunManifest '{manifest.run_id}' is missing ground_truth_path or doc_ids."
        )

    gt_key = TASK_GROUND_TRUTH_KEY.get(manifest.task, "")
    if not gt_key:
        raise ValueError(f"Unknown task '{manifest.task}' — cannot determine ground truth key.")

    if not Path(manifest.ground_truth_path).exists():
        raise FileNotFoundError(
            f"Ground truth file not found: '{manifest.ground_truth_path}'. "
            "Evaluation cannot proceed without the original ground truth file."
        )

    actual_hash = compute_ground_truth_hash(
        manifest.doc_ids, manifest.ground_truth_path, gt_key
    )

    if actual_hash != manifest.ground_truth_hash:
        raise RuntimeError(
            f"Ground truth hash MISMATCH for run '{manifest.run_id}'.\n"
            f"  Manifest recorded: {manifest.ground_truth_hash}\n"
            f"  Recomputed now:    {actual_hash}\n"
            f"The ground truth file '{manifest.ground_truth_path}' has changed since "
            "inference was run. Evaluation refused to prevent misleading scores."
        )

    logger.info(f"Ground truth hash verified OK for run '{manifest.run_id}'.")


def audit_doc_ids(pipeline_results: list[dict], manifest: RunManifest) -> None:
    """
    Verifies that every doc_id in the pipeline results is in the manifest's doc_ids.

    Logs a warning for any result doc_id not present in the manifest — this would
    indicate the results file does not match the manifest.
    """
    manifest_ids = set(manifest.doc_ids)
    for result in pipeline_results:
        doc_id = result["document"]["doc_id"]
        if doc_id not in manifest_ids:
            logger.warning(
                f"[{doc_id}] Result doc_id is NOT in manifest.doc_ids — "
                "results file may not match this manifest."
            )


# ─────────────────────────────────────────────────────
# Pairing Logic
# ─────────────────────────────────────────────────────

def build_metric_inputs(
    pipeline_results: list[dict],
    ground_truth: dict[str, str],
    task: PipelineTask,
) -> list[MetricInput]:
    """
    Pairs pipeline outputs with ground truth references by doc_id.

    This is the critical link between what the LLM produced and what
    it should have produced. Both sides are keyed by doc_id — if a
    doc_id exists in results but not in ground truth (or vice versa),
    it is skipped with a warning.

    Args:
        pipeline_results: Raw dicts from load_pipeline_results()
        ground_truth: Dict of {doc_id: reference_text}
        task: Determines which field to extract as the hypothesis

    Returns:
        List of MetricInput ready for MetricsRunner
    """
    hypothesis_field = TASK_HYPOTHESIS_FIELD[task]
    inputs: list[MetricInput] = []
    skipped = 0

    for result in pipeline_results:
        doc_id = result["document"]["doc_id"]

        # ── Get reference ──────────────────────────────────────────────
        reference = ground_truth.get(doc_id)
        if not reference:
            logger.warning(f"[{doc_id}] No ground truth found — skipping.")
            skipped += 1
            continue

        # ── Get hypothesis (LLM output) ────────────────────────────────
        hypothesis = _extract_hypothesis(result, task, hypothesis_field)
        if not hypothesis:
            logger.warning(f"[{doc_id}] No hypothesis found in '{hypothesis_field}' — skipping.")
            skipped += 1
            continue

        source = result.get("document", {}).get("raw_text", "")

        inputs.append(MetricInput(
            doc_id=doc_id,
            hypothesis=hypothesis,
            reference=reference,
            source=source,
        ))

    logger.info(f"Paired {len(inputs)} documents for evaluation (skipped={skipped})")
    return inputs


def _extract_hypothesis(result: dict, task: PipelineTask, field: str) -> str:
    """
    Extracts the LLM output text from the correct result block.

    For TRANSLATION → looks inside result['translation'][field]
    For SUMMARISATION → looks inside result['summary'][field]

    Returns empty string if the block or field is missing.
    """
    if task == PipelineTask.TRANSLATION:
        block = result.get("translation") or {}
        return block.get(field, "")

    elif task == PipelineTask.SUMMARISATION:
        block = result.get("summary") or {}
        return block.get(field, "")

    elif task == PipelineTask.FULL:
        # For FULL, try translation first then summary
        translation_block = result.get("translation") or {}
        return translation_block.get(field, "")

    return ""


# ─────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────

def aggregate_scores(
    all_scores: dict[str, list[EvaluationScore]],
) -> dict[str, dict[str, float]]:
    """
    Computes mean, min, max, and std per metric across all documents.

    Corpus-level scores (doc_id == '__corpus__') are excluded from
    per-document aggregation — they are kept as-is.

    Returns:
        {
          "bleu":      {"mean": 0.42, "min": 0.10, "max": 0.75, "std": 0.18},
          "bertscore": {"mean": 0.88, "min": 0.81, "max": 0.94, "std": 0.03},
          ...
        }
    """
    aggregate: dict[str, dict[str, float]] = {}

    for metric_name, scores in all_scores.items():
        # Exclude corpus-level scores from per-doc aggregation
        doc_scores = [s.score for s in scores if s.doc_id != "__corpus__"]

        if not doc_scores:
            continue

        aggregate[metric_name] = {
            "mean":   round(statistics.mean(doc_scores), 4),
            "median": round(statistics.median(doc_scores), 4),
            "min":    round(min(doc_scores), 4),
            "max":    round(max(doc_scores), 4),
            "std":    round(statistics.stdev(doc_scores) if len(doc_scores) > 1 else 0.0, 4),
            "n_docs": len(doc_scores),
        }

    return aggregate


# ─────────────────────────────────────────────────────
# Report Saver
# ─────────────────────────────────────────────────────

def save_report(
    report: EvaluationReport,
    output_dir: str,
    task: PipelineTask,
) -> Path:
    """
    Saves the evaluation report to a timestamped JSON file.

    Filename format: report_{task}_{model}_{timestamp}.json
    Example: report_translation_gemini_1_5_pro_20260517_143022.json
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = utc_timestamp()
    model_slug = report.model_used.replace("/", "_").replace("-", "_")
    filename = f"report_{task.value}_{model_slug}_{timestamp}.json"
    full_path = output_path / filename

    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"Evaluation report saved to {full_path}")
    return full_path


# ─────────────────────────────────────────────────────
# Main Evaluator
# ─────────────────────────────────────────────────────

class Evaluator:
    """
    Runs evaluation on pipeline outputs against ground truth references.

    Full flow:
        1. Load pipeline results JSON  (produced by PipelineOrchestrator)
        2. Load ground truth JSON      (produced by EuroParlLoader or CNNDailyMailLoader)
        3. Pair results with ground truth by doc_id → MetricInput list
        4. Run task-appropriate metrics via MetricsRunner
        5. Aggregate scores (mean, min, max, std)
        6. Save EvaluationReport to outputs/reports/

    Usage (manifest-first, recommended):
        evaluator = Evaluator(config)
        report = evaluator.run_on_manifest(
            "outputs/results/results_summarisation_...manifest.json"
        )

    Usage (explicit paths):
        evaluator = Evaluator(config)
        report = evaluator.run(
            results_path="outputs/results/results_translation_gemini_...json",
            ground_truth_path="data/processed/europarl/europarl_20docs.json",
            task=PipelineTask.TRANSLATION,
        )
    """

    def __init__(self, config: dict):
        self.config = config
        self.report_dir = config.get("paths", {}).get("reports", "outputs/reports/")
        eval_cfg = config.get("evaluation", {})
        self.bertscore_model = eval_cfg.get("bertscore_model", "microsoft/deberta-xlarge-mnli")
        self.comet_model = eval_cfg.get("comet_model", "Unbabel/wmt22-comet-da")

    def run_on_manifest(
        self,
        manifest_path: str | Path,
        *,
        skip_hash_verification: bool = False,
    ) -> EvaluationReport:
        """
        Evaluate using a RunManifest — the recommended production path.

        Loads the manifest, verifies ground truth hash, audits doc_ids,
        then runs evaluation.  Saves and prints the report.

        Args:
            manifest_path:          Path to a ``*.manifest.json`` file.
            skip_hash_verification: If True, bypasses hash check (dev/debug only).
                                    Logs a warning when used.

        Returns:
            EvaluationReport linked to the manifest's run_id.

        Raises:
            RuntimeError: if ground truth hash does not match the manifest.
            FileNotFoundError: if the results or ground truth files are missing.
        """
        manifest = load_manifest(manifest_path)
        logger.info(
            f"run_on_manifest | run_id={manifest.run_id} | task={manifest.task} | "
            f"model_key={manifest.model_key}"
        )

        if skip_hash_verification:
            warnings.warn(
                "skip_hash_verification=True — ground truth hash NOT checked. "
                "Scores may not be reproducible.",
                stacklevel=2,
            )
        else:
            verify_manifest_ground_truth(manifest)

        if not Path(manifest.results_path).exists():
            raise FileNotFoundError(
                f"Results file not found: '{manifest.results_path}'. "
                "The manifest points to a results file that no longer exists."
            )

        pipeline_results = load_pipeline_results(manifest.results_path)
        audit_doc_ids(pipeline_results, manifest)

        task = PipelineTask(manifest.task)
        report = self._evaluate_raw_results(
            pipeline_results,
            ground_truth_path=manifest.ground_truth_path,
            task=task,
            persist_report=True,
            run_id=manifest.run_id,
            manifest_path=str(manifest_path),
        )
        return report

    def run(
        self,
        results_path: str,
        ground_truth_path: str,
        task: PipelineTask,
    ) -> EvaluationReport:
        """
        Runs the full evaluation pipeline.

        Args:
            results_path:      Path to pipeline output JSON
            ground_truth_path: Path to processed dataset JSON with reference texts
            task:              PipelineTask — controls which metrics run and
                               which fields are compared

        Returns:
            EvaluationReport with per-document scores and aggregates
        """
        logger.info(f"Starting evaluation | task={task.value}")

        pipeline_results = load_pipeline_results(results_path)
        return self.run_on_results(
            pipeline_results=[PipelineResult.model_validate(r) for r in pipeline_results],
            ground_truth_path=ground_truth_path,
            task=task,
            persist_report=True,
        )

    def run_on_results(
        self,
        pipeline_results: list[PipelineResult],
        ground_truth_path: str,
        task: PipelineTask,
        *,
        persist_report: bool = False,
    ) -> EvaluationReport:
        """Evaluate in-memory pipeline results against ground truth references."""
        raw_results = [
            r.model_dump() if isinstance(r, PipelineResult) else r for r in pipeline_results
        ]
        return self._evaluate_raw_results(
            raw_results,
            ground_truth_path=ground_truth_path,
            task=task,
            persist_report=persist_report,
        )

    def _evaluate_raw_results(
        self,
        pipeline_results: list[dict],
        ground_truth_path: str,
        task: PipelineTask,
        *,
        persist_report: bool,
        run_id: str | None = None,
        manifest_path: str | None = None,
    ) -> EvaluationReport:
        ground_truth = load_ground_truth(ground_truth_path, task)
        metric_inputs = build_metric_inputs(pipeline_results, ground_truth, task)

        if not metric_inputs:
            raise ValueError(
                "No valid pairs found. Check that doc_ids match between "
                "results and ground truth, and that the correct task is set."
            )

        metrics_to_run = get_task_metrics(self.config, task)
        logger.info(f"Running metrics: {metrics_to_run}")

        runner = MetricsRunner(
            metrics=metrics_to_run,
            bertscore_model=self.bertscore_model,
            comet_model=self.comet_model,
        )
        all_scores = runner.run_all(metric_inputs)
        aggregate = aggregate_scores(all_scores)
        flat_scores = [score for scores in all_scores.values() for score in scores]
        model_used = _detect_model(pipeline_results, task)

        report = EvaluationReport(
            model_used=model_used,
            scores=flat_scores,
            aggregate=aggregate,
            run_id=run_id,
            manifest_path=manifest_path,
        )
        if persist_report:
            save_report(report, self.report_dir, task)
            self._print_summary(report, task)
        return report

    def _print_summary(self, report: EvaluationReport, task: PipelineTask) -> None:
        """Prints a clean summary table to stdout."""
        print(f"\n{'─' * 55}")
        print(f"  Evaluation Summary")
        print(f"  Task:  {task.value.upper()}")
        print(f"  Model: {report.model_used}")
        if report.run_id:
            print(f"  Run:   {report.run_id}")
        print(f"{'─' * 55}")
        print(f"  {'Metric':<15} {'Mean':>8} {'Min':>8} {'Max':>8} {'Std':>8}")
        print(f"  {'─' * 49}")
        for metric, stats in report.aggregate.items():
            print(
                f"  {metric:<15} "
                f"{stats['mean']:>8.4f} "
                f"{stats['min']:>8.4f} "
                f"{stats['max']:>8.4f} "
                f"{stats['std']:>8.4f}"
            )
        print(f"{'─' * 55}\n")


def _detect_model(pipeline_results: list[dict], task: PipelineTask) -> str:
    """Reads model_used from the first available result block."""
    for result in pipeline_results:
        if task == PipelineTask.TRANSLATION:
            block = result.get("translation") or {}
        elif task == PipelineTask.SUMMARISATION:
            block = result.get("summary") or {}
        else:
            block = result.get("translation") or result.get("summary") or {}
        model = block.get("model_used", "")
        if model:
            return model
    return "unknown"


# ─────────────────────────────────────────────────────
# Path resolution helpers
# ─────────────────────────────────────────────────────

# Maps each task to the config datasets key that holds its ground truth.
_TASK_DATASET_KEY: dict[PipelineTask, str] = {
    PipelineTask.TRANSLATION:   "europarl",
    PipelineTask.SUMMARISATION: "cnn_dailymail",
    PipelineTask.FULL:          "europarl",
}


def resolve_results_path(task: PipelineTask, outputs_dir: str) -> Path:
    """
    Returns the path stored in the latest pointer file for the given task.

    .. deprecated::
        This function supports the legacy ``--latest`` dev flow.
        Prefer ``--run <manifest_path>`` for production evaluation.

    The orchestrator writes outputs/results/latest_{task}.txt after every run,
    containing the absolute path of the most recent results JSON.

    Raises:
        FileNotFoundError: if the pointer file does not exist (no run yet).
    """
    pointer = Path(outputs_dir) / f"latest_{task.value}.txt"
    if not pointer.exists():
        raise FileNotFoundError(
            f"No latest-results pointer found at '{pointer}'. "
            f"Run the orchestrator first, or pass --run <manifest.json> explicitly."
        )
    return Path(pointer.read_text(encoding="utf-8").strip())


def resolve_ground_truth_path(task: PipelineTask, config: dict) -> str:
    """Returns the processed dataset path derived from sample_size in config."""
    dataset_key = _TASK_DATASET_KEY[task]
    return get_processed_path(config, dataset_key)


# ─────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(
        description="Run LLMEvalForge evaluation on pipeline outputs."
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--run",
        dest="manifest_path",
        metavar="MANIFEST",
        help=(
            "Path to a *.manifest.json file produced by the orchestrator. "
            "Verifies ground truth hash before scoring — recommended for all evaluations."
        ),
    )
    mode_group.add_argument(
        "--results",
        default=None,
        help="Path to pipeline results JSON (requires --ground-truth and --task).",
    )
    mode_group.add_argument(
        "--latest",
        action="store_true",
        help=(
            "[DEV ONLY] Auto-discover the latest results via the pointer file. "
            "Prints a warning. Requires --task."
        ),
    )

    parser.add_argument(
        "--ground-truth",
        default=None,
        dest="ground_truth",
        help="Path to processed dataset JSON with reference texts (required with --results).",
    )
    parser.add_argument(
        "--task",
        default=None,
        choices=[t.value for t in PipelineTask],
        help="Task type — required with --results and --latest.",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to config.yaml (default: configs/config.yaml)",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    evaluator = Evaluator(config=config)

    if args.manifest_path:
        # ── Production path: manifest-first ───────────────────────────
        report = evaluator.run_on_manifest(args.manifest_path)
        print(f"Report saved to: {config.get('paths', {}).get('reports', 'outputs/reports/')}")

    elif args.results:
        # ── Explicit paths ─────────────────────────────────────────────
        if not args.task:
            parser.error("--task is required when using --results")
        if not args.ground_truth:
            parser.error("--ground-truth is required when using --results")

        task = PipelineTask(args.task)
        report = evaluator.run(
            results_path=args.results,
            ground_truth_path=args.ground_truth,
            task=task,
        )
        print(f"Report saved to: {config.get('paths', {}).get('reports', 'outputs/reports/')}")

    else:
        # ── Dev-only: latest pointer ───────────────────────────────────
        if not args.task:
            parser.error("--task is required when using --latest")

        print(
            "\n[WARNING] --latest resolves to the most recent run via pointer file.\n"
            "          This is a DEV convenience — use --run <manifest.json> for\n"
            "          reproducible, hash-verified evaluation.\n"
        )
        task = PipelineTask(args.task)
        outputs_dir = config.get("paths", {}).get("outputs", "outputs/results/")
        results_path = str(resolve_results_path(task, outputs_dir))
        ground_truth_path = args.ground_truth or resolve_ground_truth_path(task, config)

        logger.info(f"Results (latest):  {results_path}")
        logger.info(f"Ground truth:      {ground_truth_path}")

        report = evaluator.run(
            results_path=results_path,
            ground_truth_path=ground_truth_path,
            task=task,
        )
        print(f"Report saved to: {config.get('paths', {}).get('reports', 'outputs/reports/')}")
