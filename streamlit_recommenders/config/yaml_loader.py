"""Loader for top-level YAML configuration files."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path | None) -> dict[str, Any]:
    """Load a YAML config file into a dict.

    Args:
        path: Path to the YAML file, or ``None``.

    Returns:
        The parsed mapping; an empty dict when ``path`` is ``None`` or the file
        is empty.

    Raises:
        ValueError: If the YAML top level is not a mapping.
    """
    if path is None:
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config YAML must be a mapping at the top level")
    return data
