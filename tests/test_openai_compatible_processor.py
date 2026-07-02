from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.core.models import DocumentInput
from src.core.pricing import TokenUsage, calculate_cost
from src.providers.openai_compatible_processor import OpenAICompatibleProcessor
from tests.conftest import OPENAI_CONFIG, OPENROUTER_CONFIG, SAMPLE_DOCUMENT, SAMPLE_PROMPTS


def _make_openai_response(text: str, input_tokens: int = 150, output_tokens: int = 75):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=input_tokens, completion_tokens=output_tokens),
    )


@pytest.fixture
def openai_processor():
    with patch("src.providers.openai_compatible_processor.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        processor = OpenAICompatibleProcessor(
            api_key="test-key",
            config=OPENAI_CONFIG,
            prompts=SAMPLE_PROMPTS,
        )
        processor._client = mock_client
        yield processor, mock_cls


@pytest.fixture
def openrouter_processor():
    with patch("src.providers.openai_compatible_processor.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        processor = OpenAICompatibleProcessor(
            api_key="router-key",
            config=OPENROUTER_CONFIG,
            prompts=SAMPLE_PROMPTS,
        )
        processor._client = mock_client
        yield processor, mock_cls


def test_openai_client_uses_default_base_url(openai_processor):
    _, mock_openai_cls = openai_processor
    mock_openai_cls.assert_called_once_with(api_key="test-key", base_url=None)


def test_openrouter_client_uses_custom_base_url(openrouter_processor):
    _, mock_openai_cls = openrouter_processor
    mock_openai_cls.assert_called_once_with(
        api_key="router-key",
        base_url="https://openrouter.ai/api/v1",
    )


def test_extract_parses_usage_tokens(openai_processor):
    processor, _ = openai_processor
    payload = '{"entities": ["EU"], "dates": [], "deadlines": [], "topics": [], "key_clauses": []}'
    processor._client.chat.completions.create.return_value = _make_openai_response(payload, 400, 100)

    document = DocumentInput(**SAMPLE_DOCUMENT)
    result = processor.extract(document)

    assert result.token_usage == TokenUsage(input_tokens=400, output_tokens=100)
    assert result.cost_usd == calculate_cost(result.token_usage, OPENAI_CONFIG["pricing"])


def test_parse_json_strips_markdown_fences(openai_processor):
    processor, _ = openai_processor
    fenced = '```json\n{"summary": "Meeting notes", "key_points": ["budget"], "action_items": []}\n```'
    processor._client.chat.completions.create.return_value = _make_openai_response(fenced)

    document = DocumentInput(**SAMPLE_DOCUMENT)
    result = processor.summarise(document)

    assert result.summary == "Meeting notes"
    assert result.key_points == ["budget"]
    assert result.token_usage is not None
