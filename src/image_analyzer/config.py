"""Configuration discovery and validation for the one-argument CLI."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path

DEFAULTS = {
    "overwrite": False,
    "recursive": False,
    "analysis": {
        "base_url": "http://localhost:1234/v1", "model": "qwen/qwen3-vl-8b",
        "timeout": 180, "max_tokens": 1200, "load_model": True, "unload_others": True,
    },
    "generation": {
        "diffusion_model": "", "text_encoder": "", "vae": "", "sd_cli": "sd-cli",
        "timeout": 1800, "width": 1024, "height": 1024, "steps": 40, "seed": 42,
        "cfg_scale": 6.0, "unload_lm_studio": True, "keep_original": False,
    },
}


def find_config() -> Path | None:
    """Use an environment path, cwd, user config, or editable project config."""
    explicit = os.getenv("IMAGE_ANALYZER_CONFIG")
    if explicit:
        return Path(explicit).expanduser().resolve()
    candidates = (
        Path.cwd() / "image-analyzer.json",
        Path.home() / ".config" / "image-analyzer" / "config.json",
        Path(__file__).resolve().parents[2] / "image-analyzer.json",
    )
    return next((path for path in candidates if path.is_file()), None)


def load_config(path: str | Path | None = None) -> dict:
    """Load partial JSON over defaults; asset paths are relative to the config."""
    source = Path(path).expanduser().resolve() if path is not None else find_config()
    settings = deepcopy(DEFAULTS)
    if source is None:
        return settings
    try:
        supplied = json.loads(source.read_text(encoding="utf-8"))
        _merge(settings, supplied)
    except (ValueError, OSError) as exc:
        raise ValueError(f"Invalid configuration {source}: {exc}") from exc
    generation = settings["generation"]
    for key in ("diffusion_model", "text_encoder", "vae", "sd_cli"):
        value = generation[key]
        if not value or (key == "sd_cli" and not any(c in value for c in ("/", "\\"))):
            continue
        asset = Path(value).expanduser()
        generation[key] = str((source.parent / asset).resolve() if not asset.is_absolute() else asset)
    return settings


def _merge(defaults: dict, supplied: object, prefix: str = "") -> None:
    if not isinstance(supplied, dict):
        raise ValueError(f"{prefix or 'configuration'} must be a JSON object")
    for key, value in supplied.items():
        name = f"{prefix}{key}"
        if key not in defaults:
            raise ValueError(f"Unknown setting: {name}")
        expected = defaults[key]
        if isinstance(expected, dict):
            _merge(expected, value, f"{name}.")
        elif isinstance(expected, bool):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be true or false")
            defaults[key] = value
        elif isinstance(expected, (int, float)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            if isinstance(expected, int) and not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            if key != "seed" and value <= 0:
                raise ValueError(f"{name} must be positive")
            defaults[key] = value
        else:
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a string")
            if not value.strip() and key not in ("diffusion_model", "text_encoder", "vae"):
                raise ValueError(f"{name} must not be empty")
            defaults[key] = value
