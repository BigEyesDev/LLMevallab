# src/pipeline/orchestrator.py

import json
import logging
import os
import time
import uuid
from enum import Enum
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from src.core.base_processor import BaseDocumentProcessor
from src.core.time import utc_timestamp, utc_now_iso
from src.core.models import (
    DocumentInput,
    PipelineResult,
    RunManifest,
    TruncationInfo,
    TASK_GROUND_TRUTH_KEY,
    DEFAULT_TASK_TRUNCATION_LIMITS,
)
from src.core.config import get_model_catalog, get_processed_path, load_config, validate_model_key
from src.core.provenance import dataset_hash, ground_truth_hash, config_hash, config_snapshot
from src.providers.gemini_processor import GeminiProcessor
from src.providers.claude_processor import ClaudeProcessor
from src.providers.openai_compatible_processor import OpenAICompatibleProcessor
from src.pipeline.europarl_loader import EuroParlDataLoader
from src.pipeline.cnn_dailymail_loader import CNNDailyMailLoader

load_dotenv()
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
# Task Enum — defines what the pipeline does per document
# ─────────────────────────────────────────────────────

class PipelineTask(str, Enum):
    """
    Controls which steps the orchestrator runs per document.

    TRANSLATION    → extract + translate only         (use with EuroParl)
    SUMMARISATION  → extract + summarise only          (use with CNN/DailyMail)
    FULL           → extract + translate + summarise   (use with multilingual docs
                                                        that need both)

    Why this matters:
        EuroParl documents are German text — they need translation.
        CNN/DailyMail documents are English articles — they need summarisation.
        Running translation on CNN/DailyMail is pointless (already English).
        Running summarisation on EuroParl means summarising EU parliament
        sentences, not real articles — not what you want to evaluate.
    """
    TRANSLATION = "translation"
    SUMMARISATION = "summarisation"
    FULL = "full"


# ─────────────────────────────────────────────────────
# Config & Prompt Loaders
# ─────────────────────────────────────────────────────

def load_prompts(prompts_path: str = "configs/prompts.yaml") -> dict:
    """Load prompt templates from YAML (delegates to prompt_manager)."""
    from src.pipeline.prompt_manager import load_prompts as _load

    return _load(prompts_path)


def _dataset_sample_size(config: dict, dataset_key: str) -> int:
    return config["datasets"][dataset_key]["sample_size"]


# ─────────────────────────────────────────────────────
# Processor Factory
# ─────────────────────────────────────────────────────

def build_processor(model_key: str, config: dict, prompts: dict) -> BaseDocumentProcessor:
    """
    Factory function — returns the right processor for the given catalog model key.
    """
    validate_model_key(model_key, config)
    model_config = get_model_catalog(config)[model_key]
    provider_type = model_config["provider_type"]

    if provider_type == "gemini":
        api_key = os.environ.get(model_config["api_key_env"])
        if not api_key:
            raise EnvironmentError(f"{model_config['api_key_env']} not set in environment.")
        return GeminiProcessor(api_key=api_key, config=model_config, prompts=prompts)

    if provider_type == "claude":
        api_key = os.environ.get(model_config["api_key_env"])
        if not api_key:
            raise EnvironmentError(f"{model_config['api_key_env']} not set in environment.")
        return ClaudeProcessor(api_key=api_key, config=model_config, prompts=prompts)

    if provider_type == "openai_compatible":
        api_key = os.environ.get(model_config["api_key_env"])
        if not api_key:
            raise EnvironmentError(f"{model_config['api_key_env']} not set in environment.")
        return OpenAICompatibleProcessor(api_key=api_key, config=model_config, prompts=prompts)

    raise ValueError(f"Unknown provider_type: '{provider_type}'.")


# ─────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────

class PipelineOrchestrator:
    """
    Runs the document processing pipeline.

    The steps that execute per document depend on the PipelineTask:

        PipelineTask.TRANSLATION   → extract + translate
        PipelineTask.SUMMARISATION → extract + summarise
        PipelineTask.FULL          → extract + translate + summarise

    Every run writes a timestamped results JSON **and** a companion
    ``{stem}.manifest.json`` recording dataset/ground-truth hashes,
    doc_ids, config snapshot, and the path to the results file.
    The manifest enables reproducible re-evaluation and hash-verified
    ground truth matching — see Priority 1 in PHASE_2B.md.
    """

    def __init__(
        self,
        processor: BaseDocumentProcessor,
        config: dict,
        task: PipelineTask = PipelineTask.TRANSLATION,
        *,
        model_key: str | None = None,
        dataset_path: str | None = None,
        ground_truth_path: str | None = None,
        prompt_version: str | None = None,
    ):
        """
        Args:
            processor:          Any BaseDocumentProcessor implementation.
            config:             Loaded config.yaml dict.
            task:               Controls which pipeline steps run — see PipelineTask.
            model_key:          Catalog key (e.g. 'gemini-2.5-flash') — used for
                                config hashing in the manifest.  Optional for
                                backward-compatibility; omit in BenchmarkRunner paths
                                that do not need strict manifest hash tracking.
            dataset_path:       Path to the input documents JSON — stored in
                                manifest and hashed.  Optional for backward-compat.
            ground_truth_path:  Path to the ground truth dataset JSON — stored in
                                manifest and hashed.  Defaults to ``dataset_path``
                                when both point to the same file (our current loaders).
        """
        self.processor = processor
        self.config = config
        self.task = task
        self.model_key = model_key
        self.dataset_path = dataset_path
        self.ground_truth_path = ground_truth_path or dataset_path
        self.prompt_version = prompt_version
        self.output_dir = Path(config["paths"]["outputs"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Orchestrator ready — model: {processor}, task: {task.value}")

    def run(self, documents: list[DocumentInput]) -> list[PipelineResult]:
        """
        Processes a list of documents through the configured pipeline steps.

        Args:
            documents: List of DocumentInput objects

        Returns:
            List of PipelineResult — one per document
        """
        results: list[PipelineResult] = []
        target_lang = self.config["pipeline"]["target_language"]

        logger.info(
            f"Starting [{self.task.value}] pipeline | "
            f"model: {self.processor.model_name} | "
            f"documents: {len(documents)}"
        )

        for doc in tqdm(documents, desc=f"[{self.task.value}] {self.processor.model_name}"):
            result = self._process_single(doc, target_lang)
            results.append(result)

        output_path = self._save_results(results)
        logger.info(f"Pipeline complete. Results → {output_path}")
        return results

    def _get_task_truncation_limit(self) -> int:
        """Returns the per-task character truncation limit from config, with a fallback."""
        per_task = (
            self.config.get("pipeline", {})
            .get("max_document_length_per_task", {})
        )
        task_val = self.task.value
        if task_val in per_task:
            return int(per_task[task_val])
        # Fall back to global pipeline limit, then hard-coded defaults.
        global_limit = self.config.get("pipeline", {}).get("max_document_length")
        return int(global_limit) if global_limit else DEFAULT_TASK_TRUNCATION_LIMITS.get(task_val, 2000)

    def _truncate_document(self, document: DocumentInput, limit: int) -> tuple[DocumentInput, TruncationInfo]:
        """
        Returns a (possibly truncated) document copy and the corresponding TruncationInfo.

        When the document's raw_text is within the limit, the original document is returned
        unchanged and ``was_truncated`` is False.
        """
        original_len = len(document.raw_text)
        if original_len <= limit:
            return document, TruncationInfo(
                chars_original=original_len,
                chars_sent=original_len,
                was_truncated=False,
                limit_applied=limit,
            )

        truncated_doc = document.model_copy(update={"raw_text": document.raw_text[:limit]})
        logger.debug(
            f"[{document.doc_id}] Truncated: {original_len} → {limit} chars "
            f"(task={self.task.value})"
        )
        return truncated_doc, TruncationInfo(
            chars_original=original_len,
            chars_sent=limit,
            was_truncated=True,
            limit_applied=limit,
        )

    def _process_single(self, document: DocumentInput, target_language: str) -> PipelineResult:
        """
        Runs the appropriate pipeline steps on one document based on self.task.

        Step routing:
            TRANSLATION   → extract, translate           (no summarise)
            SUMMARISATION → extract, summarise            (no translate)
            FULL          → extract, translate, summarise

        Each step is individually error-handled — a failure in one step
        does not abort the remaining steps for that document.

        Truncation is applied before any LLM step using the per-task limit from config.
        The original document (with full raw_text) is preserved in the result;
        the truncated copy is only passed to the processor.
        """
        start = time.time()

        extraction = None
        translation = None
        summary = None

        limit = self._get_task_truncation_limit()
        truncated_doc, trunc_info = self._truncate_document(document, limit)

        # ── Step 1: Extract (runs for all tasks) ──────────────────────
        try:
            extraction = self.processor.extract(truncated_doc)
        except Exception as e:
            logger.error(f"[{document.doc_id}] Extraction failed: {e}")

        # ── Step 2: Translate (TRANSLATION and FULL tasks only) ────────
        if self.task in (PipelineTask.TRANSLATION, PipelineTask.FULL):
            try:
                translation = self.processor.translate(
                    truncated_doc, target_language=target_language
                )
            except Exception as e:
                logger.error(f"[{document.doc_id}] Translation failed: {e}")

        # ── Step 3: Summarise (SUMMARISATION and FULL tasks only) ──────
        if self.task in (PipelineTask.SUMMARISATION, PipelineTask.FULL):
            try:
                # For FULL task: summarise the translated text if available
                # For SUMMARISATION task: document is already in target language
                # (e.g. CNN/DailyMail is English) — summarise raw text directly
                if self.task == PipelineTask.FULL and translation:
                    summary_input = truncated_doc.model_copy(
                        update={
                            "raw_text": translation.translated_text,
                            "source_language": target_language,
                        }
                    )
                else:
                    summary_input = truncated_doc

                summary = self.processor.summarise(summary_input)
            except Exception as e:
                logger.error(f"[{document.doc_id}] Summarisation failed: {e}")

        return PipelineResult(
            document=document,      # always store the original, untruncated document
            extraction=extraction,
            translation=translation,
            summary=summary,
            total_processing_time_ms=(time.time() - start) * 1000,
            truncation=trunc_info,
            prompt_version=self.prompt_version,
        )

    def _save_results(self, results: list[PipelineResult]) -> Path:
        """
        Saves results to a timestamped JSON file, updates the latest pointer,
        and writes a companion ``{stem}.manifest.json`` for provenance tracking.

        Filename format: results_{task}_{model}_{timestamp}.json
        Example: results_translation_gemini_1_5_pro_20260517_143022.json

        Timestamped so consecutive runs never overwrite each other.
        A companion pointer file (latest_{task}.txt) is also written so that
        downstream tools can locate the most recent run without specifying an
        explicit path (use ``--latest`` flag on the evaluator CLI).
        """
        timestamp = utc_timestamp()
        run_id = f"{timestamp}_{uuid.uuid4().hex[:6]}"
        model_name = self.processor.model_name.replace("/", "_").replace("-", "_")
        filename = f"results_{self.task.value}_{model_name}_{timestamp}.json"
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

        pointer_path = self.output_dir / f"latest_{self.task.value}.txt"
        pointer_path.write_text(str(output_path.resolve()), encoding="utf-8")
        logger.info(f"Latest pointer updated → {pointer_path}")

        self._write_manifest(results, run_id, output_path)

        return output_path

    def _write_manifest(
        self,
        results: list[PipelineResult],
        run_id: str,
        results_path: Path,
    ) -> Path | None:
        """
        Writes ``{results_stem}.manifest.json`` alongside the results file.

        Hash computation is skipped gracefully when ``dataset_path`` or
        ``model_key`` were not provided (e.g. BenchmarkRunner paths).
        In that case, hash fields are empty strings, signalling that
        hash verification is not available for this run.
        """
        from src import __version__

        doc_ids = [r.document.doc_id for r in results]
        sample_indices = list(range(len(results)))

        d_hash = ""
        gt_hash = ""
        c_hash = ""
        c_snapshot: dict = {}
        gt_path = self.ground_truth_path or ""

        if self.dataset_path and Path(self.dataset_path).exists():
            try:
                d_hash = dataset_hash(self.dataset_path)
            except Exception as e:
                logger.warning(f"dataset_hash failed: {e}")

        if self.ground_truth_path and Path(self.ground_truth_path).exists() and doc_ids:
            try:
                gt_key = TASK_GROUND_TRUTH_KEY.get(self.task.value, "")
                if gt_key:
                    gt_hash = ground_truth_hash(doc_ids, self.ground_truth_path, gt_key)
            except Exception as e:
                logger.warning(f"ground_truth_hash failed: {e}")

        if self.model_key:
            try:
                c_hash = config_hash(self.config, self.model_key)
                c_snapshot = config_snapshot(self.config, self.model_key)
                # Annotate the snapshot with the effective truncation limit for this task
                c_snapshot["truncation_limit_applied"] = self._get_task_truncation_limit()
                c_snapshot["task"] = self.task.value
            except Exception as e:
                logger.warning(f"config_hash failed: {e}")

        catalog = get_model_catalog(self.config)
        model_id = ""
        if self.model_key and self.model_key in catalog:
            model_id = catalog[self.model_key].get("model_id", "")

        manifest = RunManifest(
            run_id=run_id,
            app_version=__version__,
            task=self.task.value,
            model_key=self.model_key or "",
            model_id=model_id,
            dataset_path=self.dataset_path or "",
            dataset_hash=d_hash,
            doc_ids=doc_ids,
            sample_size=len(results),
            sample_indices=sample_indices,
            ground_truth_path=gt_path,
            ground_truth_hash=gt_hash,
            config_hash=c_hash,
            config_snapshot=c_snapshot,
            results_path=str(results_path.resolve()),
            created_at=utc_now_iso(),
        )

        manifest_path = results_path.with_suffix("").with_suffix(".manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.model_dump(), f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"RunManifest written → {manifest_path}")
        return manifest_path


# ─────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Run the LLMEvalForge document processing pipeline.")
    parser.add_argument(
        "--model",
        default=None,
        help="Model catalog key from config.yaml (default: models.default)",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=[t.value for t in PipelineTask],
        help=(
            "Pipeline task: "
            "'translation' for EuroParl (German→English), "
            "'summarisation' for CNN/DailyMail (English articles), "
            "'full' for both steps"
        ),
    )

    # Input source — mutually exclusive: catalog key or explicit path
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--dataset",
        dest="dataset_key",
        help=(
            "Dataset catalog key from config.yaml (e.g. 'cnn_dailymail', 'europarl'). "
            "Derives the input path from config and enables full hash tracking in the manifest."
        ),
    )
    input_group.add_argument(
        "--input",
        help="Explicit path to processed documents JSON. Use --dataset for full provenance.",
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=5,
        help="Number of documents to process (default: 5)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help=(
            "Chain evaluation immediately after inference using the manifest. "
            "Reads ground truth path and hashes from the manifest — no manual paths needed."
        ),
    )
    args = parser.parse_args()

    config = load_config()
    prompts = load_prompts()
    from src.pipeline.prompt_manager import get_prompt_version

    model_key = args.model or config["models"]["default"]

    processor = build_processor(model_key, config, prompts)
    task = PipelineTask(args.task)

    dataset_key_resolved: str | None = None
    input_path: str = ""

    if args.dataset_key:
        # --dataset path: resolve from catalog, enables full manifest hashing
        dataset_key_resolved = args.dataset_key
        input_path = get_processed_path(config, dataset_key_resolved)
        if not os.path.exists(input_path):
            print(f"Data not found at {input_path}. Download it first with the appropriate loader.")
            raise SystemExit(1)
    else:
        # --input path: explicit, backward-compat
        input_path = args.input
        if not os.path.exists(input_path):
            print(f"Input file not found: {input_path}")
            raise SystemExit(1)

    # Load documents using the right loader based on task
    if task == PipelineTask.TRANSLATION:
        loader = EuroParlDataLoader(sample_size=_dataset_sample_size(config, "europarl"))
    elif task == PipelineTask.SUMMARISATION:
        loader = CNNDailyMailLoader(sample_size=_dataset_sample_size(config, "cnn_dailymail"))
    else:
        loader = EuroParlDataLoader()

    documents = loader.load_from_disk(input_path)[: args.sample]
    print(f"Documents loaded: {len(documents)}")

    orchestrator = PipelineOrchestrator(
        processor=processor,
        config=config,
        task=task,
        model_key=model_key,
        dataset_path=input_path,
        ground_truth_path=input_path,  # same file for our current datasets
        prompt_version=get_prompt_version(prompts),
    )
    results = orchestrator.run(documents)

    output_dir = Path(config["paths"]["outputs"])
    pointer_path = output_dir / f"latest_{task.value}.txt"
    results_path = pointer_path.read_text().strip()
    manifest_path = results_path.replace(".json", ".manifest.json")

    print(f"\nPipeline complete.")
    print(f"   Task:      {task.value}")
    print(f"   Model:     {processor.model_name}")
    print(f"   Documents: {len(results)}")
    print(f"   Results:   {results_path}")
    print(f"   Manifest:  {manifest_path}")

    if args.evaluate:
        import warnings
        from src.evaluations.evaluator import Evaluator

        evaluator = Evaluator(config=config)
        print(f"\nRunning evaluation from manifest: {manifest_path}")
        report = evaluator.run_on_manifest(manifest_path)
        print(f"Evaluation complete. run_id={report.run_id}")
