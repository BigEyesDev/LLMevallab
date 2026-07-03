from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.core.models import DocumentInput
from src.core.pricing import TokenUsage, calculate_cost
from src.providers.gemini_processor import GeminiProcessor
from tests.conftest import GEMINI_CONFIG, SAMPLE_DOCUMENT, SAMPLE_PROMPTS


def _make_gemini_response(text: str, input_tokens: int = 100, output_tokens: int = 50):
    return SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(
            prompt_token_count=input_tokens,
            candidates_token_count=output_tokens,
        ),
    )


@pytest.fixture
def gemini_processor():
    with patch("src.providers.gemini_processor.genai.configure"), patch(
        "src.providers.gemini_processor.genai.GenerativeModel"
    ) as mock_model_cls:
        mock_client = MagicMock()
        mock_model_cls.return_value = mock_client
        processor = GeminiProcessor(
            api_key="test-key",
            config=GEMINI_CONFIG,
            prompts=SAMPLE_PROMPTS,
        )
        processor._client = mock_client
        yield processor


def test_extract_captures_token_usage_and_cost(gemini_processor):
    payload = '{"entities": ["Brussels"], "dates": [], "deadlines": [], "topics": [], "key_clauses": []}'
    gemini_processor._client.generate_content.return_value = _make_gemini_response(payload, 200, 80)

    document = DocumentInput(**SAMPLE_DOCUMENT)
    result = gemini_processor.extract(document)

    assert result.token_usage == TokenUsage(input_tokens=200, output_tokens=80)
    expected_cost = calculate_cost(result.token_usage, GEMINI_CONFIG["pricing"])
    assert result.cost_usd == expected_cost
    assert "Brussels" in result.entities


def test_call_api_retries_on_transient_failure(gemini_processor):
    payload = '{"entities": [], "dates": [], "deadlines": [], "topics": [], "key_clauses": []}'
    ok_response = _make_gemini_response(payload)
    gemini_processor._client.generate_content.side_effect = [
        RuntimeError("transient"),
        RuntimeError("transient"),
        ok_response,
    ]

    document = DocumentInput(**SAMPLE_DOCUMENT)
    result = gemini_processor.extract(document)

    assert result.token_usage is not None
    assert gemini_processor._client.generate_content.call_count == 3
