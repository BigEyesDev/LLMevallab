import pytest

from src.core.config import (
    cnn_dailymail_processed_path,
    europarl_processed_path,
    get_processed_path,
    load_config,
    validate_api_keys_documented,
    validate_model_catalog,
    validate_model_key,
)


def test_model_catalog_has_required_fields():
    config = load_config()
    validate_model_catalog(config)


def test_api_key_env_vars_documented_in_env_example():
    config = load_config()
    validate_api_keys_documented(config)


def test_validate_model_key_rejects_unknown():
    config = load_config()
    with pytest.raises(ValueError, match="Unknown model key"):
        validate_model_key("not-a-real-model", config)


def test_default_model_exists_in_catalog():
    config = load_config()
    default_key = config["models"]["default"]
    validate_model_key(default_key, config)


def test_processed_path_derived_from_sample_size():
    config = load_config()
    assert get_processed_path(config, "europarl") == europarl_processed_path(20, "de-en")
    assert get_processed_path(config, "cnn_dailymail") == cnn_dailymail_processed_path(20)


def test_processed_path_updates_when_sample_size_changes():
    config = load_config()
    config["datasets"]["europarl"]["sample_size"] = 100
    config["datasets"]["cnn_dailymail"]["sample_size"] = 50

    assert get_processed_path(config, "europarl") == "data/processed/europarl/europarl_de-en_100docs.json"
    assert get_processed_path(config, "cnn_dailymail") == "data/processed/cnn_dailymail/cnn_dailymail_50docs.json"


# ---------------------------------------------------------------------------
# Catalog expansion — Jul 2026 OpenRouter tier
# ---------------------------------------------------------------------------

_EXPECTED_OPENROUTER_MODELS = [
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "glm-5.2",
    "minimax-m3",
    "kimi-k2.6",
    "nemotron-3-ultra",
    "qwen3.7-plus",
]


def test_new_models_present_in_catalog():
    config = load_config()
    catalog = config["models"]["catalog"]
    for key in _EXPECTED_OPENROUTER_MODELS:
        assert key in catalog, f"Expected model '{key}' missing from catalog"


def test_new_models_use_openai_compatible_provider():
    config = load_config()
    catalog = config["models"]["catalog"]
    for key in _EXPECTED_OPENROUTER_MODELS:
        assert catalog[key]["provider_type"] == "openai_compatible", (
            f"Model '{key}' should use provider_type 'openai_compatible'"
        )


def test_new_models_use_openrouter():
    config = load_config()
    catalog = config["models"]["catalog"]
    for key in _EXPECTED_OPENROUTER_MODELS:
        assert catalog[key].get("base_url") == "https://openrouter.ai/api/v1", (
            f"Model '{key}' should use OpenRouter base_url"
        )


def test_new_models_use_openrouter_api_key():
    config = load_config()
    catalog = config["models"]["catalog"]
    for key in _EXPECTED_OPENROUTER_MODELS:
        assert catalog[key]["api_key_env"] == "OPENROUTER_API_KEY", (
            f"Model '{key}' should use OPENROUTER_API_KEY"
        )


def test_expanded_catalog_passes_full_validation():
    """All catalog entries (3 direct APIs + 7 OpenRouter) must pass field validation."""
    config = load_config()
    validate_model_catalog(config)
    assert len(config["models"]["catalog"]) == 10


def test_all_catalog_pricing_is_nonzero():
    """Every model must have positive input and output pricing — no zero/placeholder values."""
    config = load_config()
    catalog = config["models"]["catalog"]
    for key, entry in catalog.items():
        pricing = entry.get("pricing", {})
        assert pricing.get("input_per_1m", 0) > 0, (
            f"models.catalog.{key}.pricing.input_per_1m must be > 0"
        )
        assert pricing.get("output_per_1m", 0) > 0, (
            f"models.catalog.{key}.pricing.output_per_1m must be > 0"
        )
