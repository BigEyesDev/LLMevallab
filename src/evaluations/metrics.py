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
    source: str = ""   # Original document text (required for COMET translation scoring)


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


class COMETMetric:
    """
    COMET — Crosslingual Optimized Metric for Evaluation of Translation.

    Neural metric trained on human quality judgments. Requires source text,
    machine translation (hypothesis), and reference translation.

    Score range: typically 0.0–1.0 (higher is better)
    Typical use: translation quality (reference-based)

    Paper: https://arxiv.org/abs/2009.09036
    """

    name = "comet"

    def __init__(
        self,
        model_name: str = "Unbabel/wmt22-comet-da",
        model: object | None = None,
    ):
        self.model_name = model_name
        self._model = model

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from comet import download_model, load_from_checkpoint
        except ImportError as exc:
            raise ImportError(
                "COMET requires 'unbabel-comet'. Install with: uv add unbabel-comet"
            ) from exc
        model_path = download_model(self.model_name)
        self._model = load_from_checkpoint(model_path)
        return self._model

    def score(self, inputs: list[MetricInput]) -> list[EvaluationScore]:
        data = []
        for item in inputs:
            if not item.source:
                logger.warning(f"[{item.doc_id}] No source text for COMET — using empty string.")
            data.append({
                "src": item.source,
                "mt": item.hypothesis,
                "ref": item.reference,
            })

        logger.info(f"Computing COMET ({self.model_name}) on {len(inputs)} docs...")
        model = self._load_model()
        output = model.predict(data, batch_size=8, gpus=0)

        scores: list[EvaluationScore] = []
        for i, item in enumerate(inputs):
            scores.append(EvaluationScore(
                doc_id=item.doc_id,
                metric_name=self.name,
                score=round(float(output.scores[i]), 4),
            ))

        scores.append(EvaluationScore(
            doc_id="__corpus__",
            metric_name=f"{self.name}_corpus",
            score=round(float(output.system_score), 4),
        ))

        avg = sum(s.score for s in scores if s.doc_id != "__corpus__") / len(inputs) if inputs else 0
        logger.info(f"COMET average score: {avg:.4f} (system={output.system_score:.4f})")
        return scores


class LLMJudgeMetric:
    """
    LLM-as-Judge — uses a separate LLM to score summary quality.

    Evaluates faithfulness, completeness, and coherence on a 1-5 scale,
    normalized to 0-1 for the primary score (mean of the three dimensions).

    Requires source document text in MetricInput.source.
    Typical use: summarisation quality (reference-free or with reference in metadata)

    Tracks per-call latency and token cost in EvaluationScore.metadata.
    """

    name = "llm_judge"

    def __init__(
        self,
        config: dict,
        judge_model_key: str = "gpt-4o-mini",
        client: object | None = None,
    ):
        self.judge_model_key = judge_model_key
        self._config = config
        self._client = client

    def _get_client(self):
        if self._client is not None:
            return self._client
        from src.evaluations.judge import JudgeClient

        self._client = JudgeClient(self._config, self.judge_model_key)
        return self._client

    def score(self, inputs: list[MetricInput]) -> list[EvaluationScore]:
        from src.evaluations.judge import normalize_score

        client = self._get_client()
        scores: list[EvaluationScore] = []

        for item in inputs:
            if not item.source:
                logger.warning(f"[{item.doc_id}] No source text for LLM judge — skipping.")
                continue

            dims, _latency_ms, metadata = client.evaluate(item.source, item.hypothesis)
            normalized = {
                k: normalize_score(dims[k])
                for k in ("faithfulness", "completeness", "coherence")
            }
            overall = round(sum(normalized.values()) / 3, 4)

            scores.append(EvaluationScore(
                doc_id=item.doc_id,
                metric_name=self.name,
                score=overall,
                metadata={
                    **metadata,
                    "faithfulness": dims["faithfulness"],
                    "completeness": dims["completeness"],
                    "coherence": dims["coherence"],
                    "faithfulness_norm": normalized["faithfulness"],
                    "completeness_norm": normalized["completeness"],
                    "coherence_norm": normalized["coherence"],
                },
            ))

        if scores:
            avg = sum(s.score for s in scores) / len(scores)
            logger.info(
                f"LLM judge average score: {avg:.4f} "
                f"(model={self.judge_model_key}, n={len(scores)})"
            )
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
        "comet": COMETMetric,
        "llm_judge": LLMJudgeMetric,
    }

    def __init__(
        self,
        metrics: list[str],
        bertscore_model: str = "microsoft/deberta-xlarge-mnli",
        comet_model: str = "Unbabel/wmt22-comet-da",
        config: dict | None = None,
        judge_model_key: str = "gpt-4o-mini",
    ):
        self._metrics = []
        for name in metrics:
            if name not in self.AVAILABLE_METRICS:
                raise ValueError(f"Unknown metric: '{name}'. Available: {list(self.AVAILABLE_METRICS)}")
            if name == "bertscore":
                self._metrics.append(BERTScoreMetric(model_type=bertscore_model))
            elif name == "comet":
                self._metrics.append(COMETMetric(model_name=comet_model))
            elif name == "llm_judge":
                if config is None:
                    raise ValueError("LLM judge metric requires config dict.")
                self._metrics.append(LLMJudgeMetric(config=config, judge_model_key=judge_model_key))
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