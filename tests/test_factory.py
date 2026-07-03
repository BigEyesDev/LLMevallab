import copy
import os
from unittest.mock import patch

import pytest

from src.core.config import load_config
from src.pipeline.orchestrator import build_processor, load_prompts
from src.providers.claude_processor import ClaudeProcessor
from src.providers.gemini_processor import GeminiProcessor
from src.providers.openai_compatible_processor import OpenAICompatibleProcessor


@pytest.fixture
def config_and_prompts():
    return load_config(), load_prompts()


def test_build_processor_returns_gemini(config_and_prompts):
    config, prompts = config_and_prompts
    env = {"GEMINI_API_KEY": "test-gemini-key"}
    with patch.dict(os.environ, env, clear=False), patch(
        "src.providers.gemini_processor.genai.configure"
    ), patch("src.providers.gemini_processor.genai.GenerativeModel"):
        processor = build_processor("gemini-2.5-flash", config, prompts)
    assert isinstance(processor, GeminiProcessor)


def test_build_processor_returns_claude(config_and_prompts):
    config, prompts = config_and_prompts
    env = {"ANTHROPIC_API_KEY": "test-claude-key"}
    with patch.dict(os.environ, env, clear=False), patch(
        "src.providers.claude_processor.anthropic.Anthropic"
    ):
        processor = build_processor("claude-sonnet-4-6", config, prompts)
    assert isinstance(processor, ClaudeProcessor)


def test_build_processor_returns_openai_compatible(config_and_prompts):
    config, prompts = config_and_prompts
    env = {"OPENAI_API_KEY": "test-openai-key"}
    with patch.dict(os.environ, env, clear=False), patch(
        "src.providers.openai_compatible_processor.OpenAI"
    ):
        processor = build_processor("gpt-4o-mini", config, prompts)
    assert isinstance(processor, OpenAICompatibleProcessor)


def test_build_processor_unknown_provider_type_raises(config_and_prompts):
    config, prompts = config_and_prompts
    config = copy.deepcopy(config)
    config["models"]["catalog"]["broken-model"] = {
        "provider_type": "unknown_vendor",
        "model_id": "broken",
        "api_key_env": "BROKEN_API_KEY",
        "pricing": {"input_per_1m": 0.1, "output_per_1m": 0.1},
    }
    with pytest.raises(ValueError, match="Unknown provider_type"):
        build_processor("broken-model", config, prompts)


def test_build_processor_missing_api_key_raises(config_and_prompts):
    config, prompts = config_and_prompts
    env = {k: v for k, v in os.environ.items() if k != "GEMINI_API_KEY"}
    with patch.dict(os.environ, env, clear=True), patch(
        "src.providers.gemini_processor.genai.configure"
    ), patch("src.providers.gemini_processor.genai.GenerativeModel"):
        with pytest.raises(EnvironmentError, match="GEMINI_API_KEY"):
            build_processor("gemini-2.5-flash", config, prompts)
