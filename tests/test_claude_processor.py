from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.core.models import DocumentInput
from src.core.pricing import TokenUsage, calculate_cost
from src.providers.claude_processor import ClaudeProcessor
from tests.conftest import CLAUDE_CONFIG, SAMPLE_DOCUMENT, SAMPLE_PROMPTS


def _make_claude_response(text: str, input_tokens: int = 120, output_tokens: int = 60):
    return SimpleNamespace(
        content=[SimpleNamespace(text=text)],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


@pytest.fixture
def claude_processor():
    with patch("src.providers.claude_processor.anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        processor = ClaudeProcessor(
            api_key="test-key",
            config=CLAUDE_CONFIG,
            prompts=SAMPLE_PROMPTS,
        )
        processor._client = mock_client
        yield processor


def test_extract_parses_response_and_usage(claude_processor):
    payload = '{"entities": ["Parliament"], "dates": ["2024"], "deadlines": [], "topics": ["politics"], "key_clauses": []}'
    claude_processor._client.messages.create.return_value = _make_claude_response(payload, 300, 90)

    document = DocumentInput(**SAMPLE_DOCUMENT)
    result = claude_processor.extract(document)

    assert result.token_usage == TokenUsage(input_tokens=300, output_tokens=90)
    assert result.cost_usd == calculate_cost(result.token_usage, CLAUDE_CONFIG["pricing"])
    assert result.entities == ["Parliament"]


def test_translate_short_circuits_same_language(claude_processor):
    document = DocumentInput(**{**SAMPLE_DOCUMENT, "source_language": "en"})
    result = claude_processor.translate(document, target_language="en")

    assert result.translated_text == document.raw_text
    assert result.token_usage is None
    claude_processor._client.messages.create.assert_not_called()


def test_call_api_retries_on_transient_failure(claude_processor):
    payload = '{"entities": [], "dates": [], "deadlines": [], "topics": [], "key_clauses": []}'
    ok_response = _make_claude_response(payload)
    claude_processor._client.messages.create.side_effect = [
        RuntimeError("transient"),
        ok_response,
    ]

    document = DocumentInput(**SAMPLE_DOCUMENT)
    result = claude_processor.extract(document)

    assert result.token_usage is not None
    assert claude_processor._client.messages.create.call_count == 2
