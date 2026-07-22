#!/usr/bin/env python3
"""Evaluate today's post-noon pipeline results with memory-safe options.

Heavy metrics (COMET, BERTScore) load large PyTorch models. By default this script
reuses one model instance per metric across all runs. Use --light to skip them.

Examples:
  # Fast: BLEU/ROUGE only (~seconds per model)
  uv run python scripts/evaluate_today_runs.py --light

  # Translation only, full metrics, shared models
  uv run python scripts/evaluate_today_runs.py --task translation

  # Resume after a hang (skips models that already have a report today)
  uv run python scripts/evaluate_today_runs.py --resume
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

import yaml

from src.evaluations.evaluator import Evaluator
from src.evaluations.metrics import BERTScoreMetric, COMETMetric, LLMJudgeMetric, MetricsRunner
from src.evaluations.task_metrics import get_task_metrics
from src.pipeline.orchestrator import PipelineTask

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "outputs" / "results"
REPORTS_DIR = ROOT / "outputs" / "reports"

GROUND_TRUTH = {
    "translation": "data/processed/europarl/europarl_de-en_20docs.json",
    "summarisation": "data/processed/cnn_dailymail/cnn_dailymail_20docs.json",
}

TRANSLATION_MANIFESTS = [
    RESULTS_DIR / "results_translation_gemini_2.manifest.json",
    RESULTS_DIR / "results_translation_gpt_4o_mini_20260708_131847.manifest.json",
    RESULTS_DIR / "results_translation_deepseek_deepseek_v4_flash_20260708_132118.manifest.json",
    RESULTS_DIR / "results_translation_z_ai_glm_5.manifest.json",
    RESULTS_DIR / "results_translation_minimax_minimax_m3_20260708_132710.manifest.json",
    RESULTS_DIR / "results_translation_nvidia_nemotron_3_ultra_550b_a55b_20260708_132937.manifest.json",
]

SUMMARISATION_MANIFESTS = [
    RESULTS_DIR / "results_summarisation_gemini_2.manifest.json",
    RESULTS_DIR / "results_summarisation_deepseek_deepseek_v4_flash_20260708_111306.manifest.json",
    RESULTS_DIR / "results_summarisation_z_ai_glm_5.manifest.json",
    RESULTS_DIR / "results_summarisation_minimax_minimax_m3_20260708_113507.manifest.json",
    RESULTS_DIR / "results_summarisation_moonshotai_kimi_k2.manifest.json",
    RESULTS_DIR / "results_summarisation_nvidia_nemotron_3_ultra_550b_a55b_20260708_123009.manifest.json",
]


def _free_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _already_evaluated(run_id: str) -> bool:
    for report_path in REPORTS_DIR.glob("report_*_20260708_*.json"):
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("run_id") == run_id:
            return True
    return False


def _resolve_metrics(config: dict, task: PipelineTask, args: argparse.Namespace) -> list[str]:
    metrics = list(get_task_metrics(config, task))
    if args.light:
        return ["bleu"] if task == PipelineTask.TRANSLATION else ["rouge"]
    if args.skip_comet:
        metrics = [m for m in metrics if m != "comet"]
    if args.skip_bertscore:
        metrics = [m for m in metrics if m != "bertscore"]
    if args.skip_llm_judge:
        metrics = [m for m in metrics if m != "llm_judge"]
    return metrics


def _build_shared_runner(
    config: dict,
    task: PipelineTask,
    metrics: list[str],
    *,
    comet_batch_size: int,
) -> MetricsRunner:
    eval_cfg = config.get("evaluation", {})
    bertscore_model = eval_cfg.get("bertscore_model", "microsoft/deberta-xlarge-mnli")
    comet_model = eval_cfg.get("comet_model", "Unbabel/wmt22-comet-da")
    judge_model_key = eval_cfg.get("judge_model", "gpt-4o-mini")

    shared: dict[str, object] = {}
    if "bertscore" in metrics:
        logger.info("Loading BERTScore model once (%s)...", bertscore_model)
        shared["bertscore"] = BERTScoreMetric(model_type=bertscore_model)
    if "comet" in metrics:
        logger.info("Loading COMET model once (%s, batch_size=%d)...", comet_model, comet_batch_size)
        shared["comet"] = COMETMetric(model_name=comet_model, batch_size=comet_batch_size)
    if "llm_judge" in metrics:
        shared["llm_judge"] = LLMJudgeMetric(config=config, judge_model_key=judge_model_key)

    return MetricsRunner(
        metrics=metrics,
        bertscore_model=bertscore_model,
        comet_model=comet_model,
        config=config,
        judge_model_key=judge_model_key,
        shared_metrics=shared,
        comet_batch_size=comet_batch_size,
    )


def _evaluate_manifest(
    evaluator: Evaluator,
    manifest_path: Path,
    *,
    skip_hash: bool,
) -> None:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("ground_truth_hash") and data.get("ground_truth_path"):
        evaluator.run_on_manifest(manifest_path, skip_hash_verification=skip_hash)
        return

    task = PipelineTask(data["task"])
    evaluator.run(
        results_path=data["results_path"],
        ground_truth_path=GROUND_TRUTH[task.value],
        task=task,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=["translation", "summarisation", "all"],
        default="all",
        help="Which task batch to evaluate (default: all)",
    )
    parser.add_argument(
        "--light",
        action="store_true",
        help="Fast mode: BLEU only (translation) or ROUGE only (summarisation)",
    )
    parser.add_argument("--skip-comet", action="store_true", help="Skip COMET (saves ~2 GB RAM per reload)")
    parser.add_argument("--skip-bertscore", action="store_true", help="Skip BERTScore")
    parser.add_argument("--skip-llm-judge", action="store_true", help="Skip LLM-as-judge API calls")
    parser.add_argument(
        "--comet-batch-size",
        type=int,
        default=2,
        help="COMET batch size — lower uses less RAM (default: 2)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip models that already have a report from today (20260708)",
    )
    parser.add_argument(
        "--skip-hash",
        action="store_true",
        help="Skip ground-truth hash verification (for older manifests)",
    )
    args = parser.parse_args()

    config_path = ROOT / "configs" / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    batches: list[tuple[PipelineTask, list[Path]]] = []
    if args.task in ("translation", "all"):
        batches.append((PipelineTask.TRANSLATION, TRANSLATION_MANIFESTS))
    if args.task in ("summarisation", "all"):
        batches.append((PipelineTask.SUMMARISATION, SUMMARISATION_MANIFESTS))

    ok, skipped, failed = 0, 0, 0

    for task, manifests in batches:
        metrics = _resolve_metrics(config, task, args)
        logger.info("=== %s | metrics: %s ===", task.value.upper(), metrics)

        runner = _build_shared_runner(
            config,
            task,
            metrics,
            comet_batch_size=args.comet_batch_size,
        )
        evaluator = Evaluator(
            config,
            metrics_override=metrics,
            metrics_runner=runner,
        )

        for manifest_path in manifests:
            label = manifest_path.name
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            run_id = manifest_data.get("run_id", "")

            if args.resume and run_id and _already_evaluated(run_id):
                logger.info("SKIP (already evaluated): %s [%s]", label, run_id)
                skipped += 1
                continue

            try:
                logger.info("--- %s ---", label)
                _evaluate_manifest(evaluator, manifest_path, skip_hash=args.skip_hash)
                ok += 1
            except Exception:
                logger.exception("FAILED: %s", label)
                failed += 1
            finally:
                _free_memory()

        del runner, evaluator
        _free_memory()

    logger.info("Done: %d succeeded, %d skipped, %d failed", ok, skipped, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
