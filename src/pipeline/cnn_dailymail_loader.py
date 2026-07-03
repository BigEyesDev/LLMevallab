# src/pipeline/cnn_dailymail_loader.py

import json
import logging
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from src.core.config import cnn_dailymail_processed_path
from src.core.models import DocumentInput

logger = logging.getLogger(__name__)


class CNNDailyMailLoader:
    """
    Loads CNN/DailyMail dataset for summarisation evaluation.

    Each document contains:
        - raw_text: full news article
        - metadata.reference_summary: human-written highlight summary

    The reference_summary is used as ground truth when computing
    ROUGE-L and BERTScore on summarisation outputs.

    Dataset: https://huggingface.co/datasets/abisee/cnn_dailymail
    Version: 3.0.0 (standard benchmark version)
    """

    DATASET_NAME = "abisee/cnn_dailymail"
    VERSION = "3.0.0"

    def __init__(self, processed_dir: str = "data/processed/cnn_dailymail/", sample_size: int = 20):
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.sample_size = sample_size

    def download_and_prepare(self) -> Path:
        """Downloads CNN/DailyMail and saves a processed subset with reference summaries."""
        logger.info(f"Loading CNN/DailyMail, {self.sample_size} samples...")

        dataset = load_dataset(
            self.DATASET_NAME,
            self.VERSION,
            split="test",   # Use test split — it's the standard eval split
            trust_remote_code=True,
        )

        documents: list[dict] = []

        for i, example in enumerate(tqdm(dataset, total=self.sample_size, desc="CNN/DailyMail")):
            if i >= self.sample_size:
                break

            doc = DocumentInput(
                doc_id=f"cnn_dm_{i:04d}",
                source_language="en",
                raw_text=example["article"],
                source="cnn_dailymail",
                metadata={
                    # This is the ground truth we evaluate summaries against
                    "reference_summary": example["highlights"],
                    "cnn_id": example.get("id", ""),
                },
            )
            documents.append(doc.model_dump())

        output_path = Path(cnn_dailymail_processed_path(self.sample_size))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)

        logger.info(f"CNN/DailyMail: saved {len(documents)} documents to {output_path}")
        return output_path

    def load_from_disk(self, file_path: str) -> list[DocumentInput]:
        with open(file_path, "r", encoding="utf-8") as f:
            return [DocumentInput(**doc) for doc in json.load(f)]

    def get_reference_summaries(self, file_path: str) -> dict[str, str]:
        """Returns {doc_id: reference_summary} for evaluation."""
        return {
            doc.doc_id: doc.metadata.get("reference_summary", "")
            for doc in self.load_from_disk(file_path)
        }