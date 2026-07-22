"""Task → metric list resolution (config + defaults)."""

from src.pipeline.orchestrator import PipelineTask

# Which metrics apply to which task.
# This prevents running BLEU on summaries or ROUGE on translations —
# each metric is only meaningful in the right context.
TASK_METRICS: dict[PipelineTask, list[str]] = {
    PipelineTask.TRANSLATION:   ["bleu", "bertscore", "comet"],
    PipelineTask.SUMMARISATION: ["rouge", "bertscore", "llm_judge"],
    PipelineTask.FULL:          ["bleu", "rouge", "bertscore", "comet", "llm_judge"],
}


def get_task_metrics(config: dict, task: PipelineTask) -> list[str]:
    """Return metrics for *task* from config, falling back to TASK_METRICS."""
    metrics_cfg = config.get("evaluation", {}).get("metrics", {})
    if task.value in metrics_cfg:
        return metrics_cfg[task.value]
    return TASK_METRICS[task]


def task_uses_llm_judge(config: dict, task: str | PipelineTask) -> bool:
    """True when *task* is configured to run the llm_judge metric."""
    if isinstance(task, str):
        task = PipelineTask(task)
    return "llm_judge" in get_task_metrics(config, task)


# Re-export for callers that import judge helpers from task_metrics.
from src.evaluations.evaluation_config import (  # noqa: E402
    apply_judge_model_override,
    get_judge_model_key,
)
