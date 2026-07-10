from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MANIFEST_NAME = "dataset.json"


def manifest_path(root: str | Path) -> Path:
    return Path(root) / MANIFEST_NAME


def read_manifest(root: str | Path) -> dict[str, Any]:
    path = manifest_path(root)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def write_manifest(root: str | Path, **fields: Any) -> dict[str, Any]:
    manifest = {"complete": True, **fields}
    manifest_path(root).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def is_complete(root: str | Path) -> bool:
    """True when a dataset folder has been fully prepared (guards re-collection)."""
    return bool(read_manifest(root).get("complete", False))
