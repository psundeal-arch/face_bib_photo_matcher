"""Shared configuration loading for downloader and website."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config file.

    Returns an empty dict when file is missing or invalid.
    """
    if not config_path.exists() or not config_path.is_file():
        return {}

    try:
        import yaml  # type: ignore
    except Exception:
        print(
            f"WARN: config file found at {config_path}, but PyYAML is not installed; ignoring config file."
        )
        return {}

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"WARN: failed to parse config file {config_path}: {exc}")
        return {}

    if isinstance(raw, dict):
        return raw
    return {}


def get_section(config: Dict[str, Any], section: str) -> Dict[str, Any]:
    value = config.get(section)
    return value if isinstance(value, dict) else {}

