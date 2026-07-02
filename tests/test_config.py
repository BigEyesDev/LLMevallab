import pytest

from src.core.config import (
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


def test_dataset_sample_sizes_match_processed_files():
    config = load_config()
    assert config["datasets"]["europarl"]["sample_size"] == 20
    assert config["datasets"]["cnn_dailymail"]["sample_size"] == 20
    assert "20docs" in config["datasets"]["europarl"]["processed_path"]
    assert "20docs" in config["datasets"]["cnn_dailymail"]["processed_path"]
