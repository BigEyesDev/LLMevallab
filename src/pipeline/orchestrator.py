# src/pipeline/orchestrator.py

import json
import logging
import os
import time
from enum import Enum
from pathlib import Path
from datetime import datetime

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from src.core.base_processor import BaseDocumentProcessor
from src.core.models import DocumentInput, PipelineResult
from src.core.config import get_model_catalog, get_processed_path, load_config, validate_model_key
from src.providers.gemini_processor import GeminiProcessor
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
    with open(prompts_path, "r") as f:
        return yaml.safe_load(f)


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
        raise NotImplementedError("Claude processor not implemented yet (Phase 2).")

    if provider_type == "openai_compatible":
        raise NotImplementedError("OpenAI-compatible processor not implemented yet (Phase 2).")

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

    This ensures EuroParl documents are only translated (not pointlessly
    summarised as EU parliament sentences), and CNN/DailyMail documents
    are only summarised (not translated — they're already English).

    Results are saved as timestamped JSON for downstream evaluation.
    """

    def __init__(
        self,
        processor: BaseDocumentProcessor,
        config: dict,
        task: PipelineTask = PipelineTask.TRANSLATION,
    ):
        """
        Args:
            processor: Any BaseDocumentProcessor implementation (Gemini, Claude, etc.)
            config: Loaded config.yaml dict
            task: Controls which pipeline steps run — see PipelineTask
        """
        self.processor = processor
        self.config = config
        self.task = task
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

    def _process_single(self, document: DocumentInput, target_language: str) -> PipelineResult:
        """
        Runs the appropriate pipeline steps on one document based on self.task.

        Step routing:
            TRANSLATION   → extract, translate           (no summarise)
            SUMMARISATION → extract, summarise            (no translate)
            FULL          → extract, translate, summarise

        Each step is individually error-handled — a failure in one step
        does not abort the remaining steps for that document.
        """
        start = time.time()

        extraction = None
        translation = None
        summary = None

        # ── Step 1: Extract (runs for all tasks) ──────────────────────
        try:
            extraction = self.processor.extract(document)
        except Exception as e:
            logger.error(f"[{document.doc_id}] Extraction failed: {e}")

        # ── Step 2: Translate (TRANSLATION and FULL tasks only) ────────
        if self.task in (PipelineTask.TRANSLATION, PipelineTask.FULL):
            try:
                translation = self.processor.translate(
                    document, target_language=target_language
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
                    summary_input = document.model_copy(
                        update={
                            "raw_text": translation.translated_text,
                            "source_language": target_language,
                        }
                    )
                else:
                    summary_input = document

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
        """
        Saves results to a timestamped JSON file and updates the latest pointer.

        Filename format: results_{task}_{model}_{timestamp}.json
        Example: results_translation_gemini_1_5_pro_20260517_143022.json

        Timestamped so consecutive runs never overwrite each other.
        A companion pointer file (latest_{task}.txt) is also written so that
        downstream tools can always locate the most recent run without
        specifying an explicit path.
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
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

        return output_path


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
    parser.add_argument(
        "--input",
        required=True,
        help="Path to processed documents JSON (from europarl_loader or cnn_dailymail_loader)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=5,
        help="Number of documents to process (default: 5)",
    )
    args = parser.parse_args()

    config = load_config()
    prompts = load_prompts()
    model_key = args.model or config["models"]["default"]

    processor = build_processor(model_key, config, prompts)
    task = PipelineTask(args.task)

    # Load documents using the right loader based on task
    if task == PipelineTask.TRANSLATION:
        loader = EuroParlDataLoader(sample_size=_dataset_sample_size(config, "europarl"))
        default_input = get_processed_path(config, "europarl")
        if not os.path.exists(args.input):
            print(f"Data not found at {args.input}, downloading...")
            loader.download_and_prepare()
            args.input = default_input
        else:
            print(f"Data found at {args.input}")
        documents = loader.load_from_disk(args.input)[: args.sample]
        print(f"Documents loaded: {len(documents)}")

    elif task == PipelineTask.SUMMARISATION:
        loader = CNNDailyMailLoader(sample_size=_dataset_sample_size(config, "cnn_dailymail"))
        default_input = get_processed_path(config, "cnn_dailymail")
        if not os.path.exists(args.input):
            print(f"Data not found at {args.input}, downloading...")
            loader.download_and_prepare()
            args.input = default_input
        else:
            print(f"Data found at {args.input}")
        documents = loader.load_from_disk(args.input)[: args.sample]
        print(f"Documents loaded: {len(documents)}")

    else:  # FULL — user passes their own multilingual documents
        loader = EuroParlDataLoader()
        documents = loader.load_from_disk(args.input)[: args.sample]

    orchestrator = PipelineOrchestrator(processor=processor, config=config, task=task)
    results = orchestrator.run(documents)

    pointer_path = Path(config["paths"]["outputs"]) / f"latest_{task.value}.txt"
    print(f"\nPipeline complete.")
    print(f"   Task:      {task.value}")
    print(f"   Model:     {processor.model_name}")
    print(f"   Documents: {len(results)}")
    print(f"   Results:   {pointer_path.read_text().strip()}")
