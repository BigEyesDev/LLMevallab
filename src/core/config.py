"""Config loading and validation helpers."""

from pathlib import Path

import yaml

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
