from abc import ABC, abstractmethod
from src.core.models import DocumentInput, ExtractionResult, TranslationResult, SummaryResult


class BaseDocumentProcessor(ABC):
    """
    Abstract base class for all LLM document processors.

    All providers (Gemini, Claude, OpenAI, open-source) must inherit
    from this class and implement all three abstract methods.

    This ensures the pipeline is fully model-agnostic — swapping models
    requires zero changes to orchestration code.

    Usage:
        class GeminiProcessor(BaseDocumentProcessor):
            def extract(self, document: DocumentInput) -> ExtractionResult:
                ...
    """

    def __init__(self, model_name: str, config: dict):
        """
        Args:
            model_name: Human-readable model identifier (e.g. 'gemini-1.5-pro')
            config: Model-specific configuration dict from config.yaml
        """
        self.model_name = model_name
        self.config = config

    @abstractmethod
    def extract(self, document: DocumentInput) -> ExtractionResult:
        """
        Extract structured information from the document.

        Must return an ExtractionResult with entities, dates,
        deadlines, topics, and key clauses populated.
        """
        ...

    @abstractmethod
    def translate(self, document: DocumentInput, target_language: str = "en") -> TranslationResult:
        """
        Translate the document text to target_language.

        If the document is already in target_language, return
        a TranslationResult with translated_text == original_text.
        """
        ...

    @abstractmethod
    def summarise(self, document: DocumentInput) -> SummaryResult:
        """
        Produce a structured summary with key points and action items.

        Summary should always be in English regardless of source language.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name})"