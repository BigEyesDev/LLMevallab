"""Multi-model benchmark runner — quality, tokens, cost, and latency side-by-side."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from src import __version__
from src.core.config import get_model_catalog, get_processed_path, load_config, validate_model_key
from src.core.models import (
    BenchmarkReport,
    DocumentInput,
    EvaluationReport,
    ModelBenchmarkResult,
    PipelineResult,
)
from src.evaluations.evaluator import Evaluator, resolve_ground_truth_path
from src.pipeline.cnn_dailymail_loader import CNNDailyMailLoader
from src.pipeline.europarl_loader import EuroParlDataLoader
from src.pipeline.orchestrator import PipelineOrchestrator, PipelineTask, build_processor, load_prompts

logger = logging.getLogger(__name__)


def load_documents_for_task(
    task: PipelineTask,
    config: dict,
    sample_size: int,
) -> list[DocumentInput]:
    """Load a fixed document slice for fair cross-model comparison."""
    if task in (PipelineTask.TRANSLATION, PipelineTask.FULL):
        dataset_key = "europarl"
        loader = EuroParlDataLoader(
            sample_size=config["datasets"]["europarl"]["sample_size"],
        )
    elif task == PipelineTask.SUMMARISATION:
        dataset_key = "cnn_dailymail"
        loader = CNNDailyMailLoader(
            sample_size=config["datasets"]["cnn_dailymail"]["sample_size"],
        )
    else:
        raise ValueError(f"Unsupported task for document loading: {task}")

    path = get_processed_path(config, dataset_key)
    documents = loader.load_from_disk(path)
    return documents[:sample_size]


def aggregate_pipeline_results(
    model_key: str,
    model_id: str,
    pipeline_results: list[PipelineResult],
    evaluation_report: EvaluationReport,
) -> ModelBenchmarkResult:
    """Combine quality metrics with token, cost, and latency aggregates."""
    input_tokens: list[int] = []
    output_tokens: list[int] = []
    total_cost = 0.0
    latencies: list[float] = []

    for result in pipeline_results:
        total_cost += _pipeline_result_cost(result)
        latencies.append(result.total_processing_time_ms)
        for step in (result.extraction, result.translation, result.summary):
            if step and step.token_usage:
                input_tokens.append(step.token_usage.input_tokens)
                output_tokens.append(step.token_usage.output_tokens)

    quality_metrics = {
        metric: stats["mean"] for metric, stats in evaluation_report.aggregate.items()
    }

    return ModelBenchmarkResult(
        model_key=model_key,
        model_id=model_id,
        quality_metrics=quality_metrics,
        avg_input_tokens=statistics.mean(input_tokens) if input_tokens else 0.0,
        avg_output_tokens=statistics.mean(output_tokens) if output_tokens else 0.0,
        total_cost_usd=round(total_cost, 8),
        avg_latency_ms=statistics.mean(latencies) if latencies else 0.0,
        n_docs=len(pipeline_results),
    )


def _pipeline_result_cost(result: PipelineResult) -> float:
    total = 0.0
    for step in (result.extraction, result.translation, result.summary):
        if step:
            total += step.cost_usd
    return total


def save_benchmark_report(
    report: BenchmarkReport,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Persist benchmark report as JSON and a flattened CSV comparison table."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    json_path = out / f"benchmark_{report.task}_{timestamp}.json"
    csv_path = out / f"benchmark_{report.task}_{timestamp}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, ensure_ascii=False, indent=2, default=str)

    rows: list[dict] = []
    for model_result in report.results:
        row = {
            "model_key": model_result.model_key,
            "model_id": model_result.model_id,
            "avg_input_tokens": model_result.avg_input_tokens,
            "avg_output_tokens": model_result.avg_output_tokens,
            "total_cost_usd": model_result.total_cost_usd,
            "avg_latency_ms": model_result.avg_latency_ms,
            "n_docs": model_result.n_docs,
        }
        for metric, value in model_result.quality_metrics.items():
            row[f"quality_{metric}"] = value
        rows.append(row)

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info("Benchmark report saved to %s and %s", json_path, csv_path)
    return json_path, csv_path


def print_benchmark_table(report: BenchmarkReport) -> None:
    """Print a side-by-side comparison table to stdout."""
    if not report.results:
        print("No benchmark results to display.")
        return

    metric_names = sorted(
        {name for result in report.results for name in result.quality_metrics}
    )
    headers = ["Model", *metric_names, "InTok", "OutTok", "Cost($)", "Latency(ms)"]

    print(f"\n{'─' * 90}")
    print(f"  Benchmark Comparison — task: {report.task} | sample: {report.sample_size}")
    print(f"{'─' * 90}")
    print("  " + " | ".join(f"{h:<14}" for h in headers))
    print(f"  {'─' * 86}")

    for result in report.results:
        cells = [result.model_key[:14]]
        for metric in metric_names:
            value = result.quality_metrics.get(metric, float("nan"))
            cells.append(f"{value:.4f}" if metric in result.quality_metrics else "n/a")
        cells.extend([
            f"{result.avg_input_tokens:.0f}",
            f"{result.avg_output_tokens:.0f}",
            f"{result.total_cost_usd:.4f}",
            f"{result.avg_latency_ms:.0f}",
        ])
        print("  " + " | ".join(f"{c:<14}" for c in cells))
    print(f"{'─' * 90}\n")


def parse_models(models_arg: str) -> list[str]:
    """Parse comma-separated model catalog keys."""
    return [key.strip() for key in models_arg.split(",") if key.strip()]


class BenchmarkRunner:
    """Run the same document sample through multiple models and compare outcomes."""

    def __init__(self, config: dict, prompts: dict):
        self.config = config
        self.prompts = prompts
        self.reports_dir = config.get("paths", {}).get("reports", "outputs/reports/")

    def run(
        self,
        task: str | PipelineTask,
        model_keys: list[str],
        sample_size: int,
    ) -> BenchmarkReport:
        """Benchmark each model on an identical document slice."""
        if isinstance(task, str):
            task = PipelineTask(task)

        for model_key in model_keys:
            validate_model_key(model_key, self.config)

        documents = load_documents_for_task(task, self.config, sample_size)
        ground_truth_path = resolve_ground_truth_path(task, self.config)
        catalog = get_model_catalog(self.config)
        evaluator = Evaluator(self.config)

        model_results: list[ModelBenchmarkResult] = []
        for model_key in model_keys:
            logger.info("Benchmarking model: %s", model_key)
            processor = build_processor(model_key, self.config, self.prompts)
            orchestrator = PipelineOrchestrator(
                processor=processor,
                config=self.config,
                task=task,
            )
            pipeline_results = orchestrator.run(documents)
            evaluation_report = evaluator.run_on_results(
                pipeline_results=pipeline_results,
                ground_truth_path=ground_truth_path,
                task=task,
                save_report=False,
            )
            model_results.append(
                aggregate_pipeline_results(
                    model_key=model_key,
                    model_id=catalog[model_key]["model_id"],
                    pipeline_results=pipeline_results,
                    evaluation_report=evaluation_report,
                )
            )

        return BenchmarkReport(
            task=task.value,
            sample_size=len(documents),
            results=model_results,
        )


def main(argv: list[str] | None = None) -> BenchmarkReport:
    """CLI entry point for multi-model benchmarking."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Run a multi-model LLM benchmark.")
    parser.add_argument(
        "--task",
        required=True,
        choices=[t.value for t in PipelineTask],
        help="Pipeline task to benchmark",
    )
    parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated model catalog keys",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=5,
        help="Number of documents to benchmark (default: 5)",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to config.yaml",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    prompts = load_prompts()
    model_keys = parse_models(args.models)

    for model_key in model_keys:
        validate_model_key(model_key, config)

    runner = BenchmarkRunner(config=config, prompts=prompts)
    report = runner.run(task=args.task, model_keys=model_keys, sample_size=args.sample)

    save_benchmark_report(report, runner.reports_dir)
    print_benchmark_table(report)
    print(f"LLMevallab v{__version__}")
    return report


if __name__ == "__main__":
    main()
