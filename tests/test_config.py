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
