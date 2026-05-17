# src/evaluation/metrics.py

import logging
from dataclasses import dataclass

import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn

from src.core.models import EvaluationScore

logger = logging.getLogger(__name__)

# Download NLTK tokeniser data if not present
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


@dataclass
class MetricInput:
    """Holds hypothesis and reference for one document."""
    doc_id: str
    hypothesis: str    # LLM output (what we're evaluating)
    reference: str     # Ground truth (what it should be)


class BLEUMetric:
    """
    BLEU (Bilingual Evaluation Understudy) Score.

    Originally designed for machine translation evaluation (Papineni et al., 2002).
    Measures how much of the LLM's output appears in the reference, using
    n-gram overlap (1-gram to 4-gram).

    Score range: 0.0 (no overlap) to 1.0 (perfect match)
    Typical use: translation quality

    Paper: https://aclanthology.org/P02-1040/
    """

    name = "bleu"

    def score(self, inputs: list[MetricInput]) -> list[EvaluationScore]:
        """
        Computes sentence-level BLEU for each input, plus corpus-level BLEU.

        Args:
            inputs: List of hypothesis/reference pairs

        Returns:
            List of EvaluationScore — one per document, plus one corpus-level score
        """
        scores: list[EvaluationScore] = []
        smoothing = SmoothingFunction().method1  # avoids zero for short texts

        for item in inputs:
            hypothesis_tokens = item.hypothesis.lower().split()
            reference_tokens = [item.reference.lower().split()]  # BLEU expects list of refs

            sentence_score = sentence_bleu(
                references=reference_tokens,
                hypothesis=hypothesis_tokens,
                smoothing_function=smoothing,
            )

            scores.append(EvaluationScore(
                doc_id=item.doc_id,
                metric_name=self.name,
                score=round(sentence_score, 4),
            ))

        # Corpus-level BLEU (more stable than sentence average)
        all_refs = [[item.reference.lower().split()] for item in inputs]
        all_hyps = [item.hypothesis.lower().split() for item in inputs]
        corpus_score = corpus_bleu(all_refs, all_hyps, smoothing_function=smoothing)

        scores.append(EvaluationScore(
            doc_id="__corpus__",
            metric_name=f"{self.name}_corpus",
            score=round(corpus_score, 4),
        ))

        logger.info(f"BLEU corpus score: {corpus_score:.4f}")
        return scores


class ROUGEMetric:
    """
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation).

    Designed for summarisation evaluation (Lin, 2004).
    We use ROUGE-L which measures the longest common subsequence —
    it captures sentence-level structure, not just n-gram matches.

    Score range: 0.0 to 1.0
    Typical use: summarisation quality

    Paper: https://aclanthology.org/W04-1013/
    """

    name = "rouge_l"

    def __init__(self):
        self._scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def score(self, inputs: list[MetricInput]) -> list[EvaluationScore]:
        scores: list[EvaluationScore] = []

        for item in inputs:
            result = self._scorer.score(
                target=item.reference,
                prediction=item.hypothesis,
            )
            rouge_l = result["rougeL"]

            scores.append(EvaluationScore(
                doc_id=item.doc_id,
                metric_name=self.name,
                score=round(rouge_l.fmeasure, 4),
                metadata={
                    "precision": round(rouge_l.precision, 4),
                    "recall": round(rouge_l.recall, 4),
                    "f1": round(rouge_l.fmeasure, 4),
                },
            ))

        avg = sum(s.score for s in scores) / len(scores) if scores else 0
        logger.info(f"ROUGE-L average F1: {avg:.4f}")
        return scores


class BERTScoreMetric:
    """
    BERTScore — Semantic Similarity via Contextual Embeddings.

    Instead of counting matching words, BERTScore encodes both hypothesis
    and reference using a pretrained transformer (DeBERTa), then computes
    cosine similarity between token embeddings.

    This means semantically equivalent sentences score high even if they
    use completely different words — a key advantage over BLEU/ROUGE.

    Score range: typically 0.80–1.0 for English (due to embedding similarity floor)
    Typical use: translation AND summarisation

    Paper: https://arxiv.org/abs/1904.09675
    """

    name = "bertscore"

    def __init__(self, model_type: str = "microsoft/deberta-xlarge-mnli", device: str = "cpu"):
        self.model_type = model_type
        self.device = device

    def score(self, inputs: list[MetricInput]) -> list[EvaluationScore]:
        hypotheses = [item.hypothesis for item in inputs]
        references = [item.reference for item in inputs]

        logger.info(f"Computing BERTScore with {self.model_type} on {len(inputs)} docs...")

        # bert_score returns tensors of P, R, F1 per sentence
        precisions, recalls, f1s = bert_score_fn(
            cands=hypotheses,
            refs=references,
            model_type=self.model_type,
            device=self.device,
            verbose=False,
        )

        scores: list[EvaluationScore] = []
        for i, item in enumerate(inputs):
            scores.append(EvaluationScore(
                doc_id=item.doc_id,
                metric_name=self.name,
                score=round(f1s[i].item(), 4),
                metadata={
                    "precision": round(precisions[i].item(), 4),
                    "recall": round(recalls[i].item(), 4),
                },
            ))

        avg = sum(s.score for s in scores) / len(scores) if scores else 0
        logger.info(f"BERTScore average F1: {avg:.4f}")
        return scores


class MetricsRunner:
    """
    Convenience class that runs all configured metrics in one call.

    Usage:
        runner = MetricsRunner(metrics=["bleu", "rouge", "bertscore"])
        all_scores = runner.run_all(inputs)
    """

    AVAILABLE_METRICS = {
        "bleu": BLEUMetric,
        "rouge": ROUGEMetric,
        "bertscore": BERTScoreMetric,
    }

    def __init__(self, metrics: list[str], bertscore_model: str = "microsoft/deberta-xlarge-mnli"):
        self._metrics = []
        for name in metrics:
            if name not in self.AVAILABLE_METRICS:
                raise ValueError(f"Unknown metric: '{name}'. Available: {list(self.AVAILABLE_METRICS)}")
            if name == "bertscore":
                self._metrics.append(BERTScoreMetric(model_type=bertscore_model))
            else:
                self._metrics.append(self.AVAILABLE_METRICS[name]())

    def run_all(self, inputs: list[MetricInput]) -> dict[str, list[EvaluationScore]]:
        """
        Runs all configured metrics.

        Returns:
            Dict mapping metric_name → list of EvaluationScore
        """
        return {
            metric.name: metric.score(inputs)
            for metric in self._metrics
        }