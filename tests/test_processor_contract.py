from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.core.base_processor import BaseDocumentProcessor
from src.core.models import DocumentInput, ExtractionResult, SummaryResult, TranslationResult
from src.providers.claude_processor import ClaudeProcessor
from src.providers.gemini_processor import GeminiProcessor
from src.providers.openai_compatible_processor import OpenAICompatibleProcessor
from tests.conftest import CLAUDE_CONFIG, GEMINI_CONFIG, OPENAI_CONFIG, SAMPLE_DOCUMENT, SAMPLE_PROMPTS


def _gemini_response(text: str):
    return SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=5),
    )


def _claude_response(text: str):
    return SimpleNamespace(
        content=[SimpleNamespace(text=text)],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def _openai_response(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


PROCESSOR_CASES = [
    ("gemini", GeminiProcessor, GEMINI_CONFIG, "src.providers.gemini_processor"),
    ("claude", ClaudeProcessor, CLAUDE_CONFIG, "src.providers.claude_processor"),
    ("openai_compatible", OpenAICompatibleProcessor, OPENAI_CONFIG, "src.providers.openai_compatible_processor"),
]


@pytest.mark.parametrize("name,processor_cls,config,module_path", PROCESSOR_CASES)
def test_processor_contract(name, processor_cls, config, module_path):
    extract_json = '{"entities": ["A"], "dates": [], "deadlines": [], "topics": [], "key_clauses": []}'
    summary_json = '{"summary": "S", "key_points": ["P"], "action_items": []}'
    translate_text = "Translated text"
    document = DocumentInput(**SAMPLE_DOCUMENT)

    if name == "gemini":
        with patch(f"{module_path}.genai.configure"), patch(f"{module_path}.genai.GenerativeModel"):
            processor = processor_cls(api_key="key", config=config, prompts=SAMPLE_PROMPTS)
            processor._client = MagicMock()
            processor._client.generate_content.side_effect = [
                _gemini_response(extract_json),
                _gemini_response(translate_text),
                _gemini_response(summary_json),
            ]
    elif name == "claude":
        with patch(f"{module_path}.anthropic.Anthropic"):
            processor = processor_cls(api_key="key", config=config, prompts=SAMPLE_PROMPTS)
            processor._client = MagicMock()
            processor._client.messages.create.side_effect = [
                _claude_response(extract_json),
                _claude_response(translate_text),
                _claude_response(summary_json),
            ]
    else:
        with patch(f"{module_path}.OpenAI"):
            processor = processor_cls(api_key="key", config=config, prompts=SAMPLE_PROMPTS)
            processor._client = MagicMock()
            processor._client.chat.completions.create.side_effect = [
                _openai_response(extract_json),
                _openai_response(translate_text),
                _openai_response(summary_json),
            ]

    assert isinstance(processor, BaseDocumentProcessor)

    extraction = processor.extract(document)
    translation = processor.translate(document, target_language="en")
    summary = processor.summarise(document)

    assert isinstance(extraction, ExtractionResult)
    assert isinstance(translation, TranslationResult)
    assert isinstance(summary, SummaryResult)

    assert extraction.token_usage is not None
    assert translation.token_usage is not None
    assert summary.token_usage is not None
