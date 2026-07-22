"""Config loading and validation helpers."""

from pathlib import Path

import yaml

from src.core.concurrency import ConcurrencySettings

REQUIRED_CATALOG_FIELDS = ("provider_type", "model_id", "api_key_env", "pricing")
REQUIRED_PRICING_FIELDS = ("input_per_1m", "output_per_1m")


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_model_catalog(config: dict) -> dict:
    try:
        return config["models"]["catalog"]
    except KeyError as exc:
        raise KeyError("config.yaml is missing models.catalog") from exc


def validate_model_catalog(config: dict) -> None:
    """Raises ValueError if any catalog entry is missing required fields."""
    catalog = get_model_catalog(config)
    if not catalog:
        raise ValueError("models.catalog must not be empty")

    for model_key, entry in catalog.items():
        for field in REQUIRED_CATALOG_FIELDS:
            if field not in entry:
                raise ValueError(f"models.catalog.{model_key} missing required field '{field}'")

        pricing = entry["pricing"]
        for field in REQUIRED_PRICING_FIELDS:
            if field not in pricing:
                raise ValueError(
                    f"models.catalog.{model_key}.pricing missing required field '{field}'"
                )


def validate_api_keys_documented(config: dict, env_example_path: str = ".env.example") -> None:
    """Raises ValueError if a catalog api_key_env is not mentioned in .env.example."""
    catalog = get_model_catalog(config)
    env_keys = {line.split("=", 1)[0].strip() for line in Path(env_example_path).read_text().splitlines()
                if "=" in line and not line.strip().startswith("#")}

    required = {entry["api_key_env"] for entry in catalog.values()}
    missing = required - env_keys
    if missing:
        raise ValueError(f".env.example missing api_key_env entries: {sorted(missing)}")


def validate_model_key(model_key: str, config: dict) -> None:
    catalog = get_model_catalog(config)
    if model_key not in catalog:
        valid = ", ".join(sorted(catalog))
        raise ValueError(f"Unknown model key '{model_key}'. Valid keys: {valid}")


def get_dataset_config(config: dict, dataset_key: str) -> dict:
    try:
        return config["datasets"][dataset_key]
    except KeyError as exc:
        raise KeyError(f"config.yaml is missing datasets.{dataset_key}") from exc


def europarl_processed_path(sample_size: int, language_pair: str = "de-en") -> str:
    return f"data/processed/europarl/europarl_{language_pair}_{sample_size}docs.json"


def cnn_dailymail_processed_path(sample_size: int) -> str:
    return f"data/processed/cnn_dailymail/cnn_dailymail_{sample_size}docs.json"


def get_concurrency_settings(config: dict) -> ConcurrencySettings:
    """Parse concurrency knobs from config.yaml."""
    pipeline = config.get("pipeline", {})
    benchmark = config.get("benchmark", {})
    evaluation = config.get("evaluation", {})
    return ConcurrencySettings(
        max_concurrent_documents=int(pipeline.get("max_concurrent_documents", 1)),
        skip_extraction=bool(pipeline.get("skip_extraction", False)),
        max_concurrent_models=int(benchmark.get("max_concurrent_models", 1)),
        max_concurrent_judge_calls=int(evaluation.get("max_concurrent_judge_calls", 1)),
        provider_limits=evaluation.get("provider_limits"),
    )


def get_processed_path(config: dict, dataset_key: str) -> str:
    """Derive processed dataset JSON path from sample_size (and language_pair for EuroParl)."""
    ds = get_dataset_config(config, dataset_key)
    sample_size = ds["sample_size"]

    if dataset_key == "europarl":
        return europarl_processed_path(sample_size, ds["language_pair"])
    if dataset_key == "cnn_dailymail":
        return cnn_dailymail_processed_path(sample_size)

    raise ValueError(f"Unknown dataset key: '{dataset_key}'")
