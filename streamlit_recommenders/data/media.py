from __future__ import annotations

from pathlib import Path

import pandas as pd

from streamlit_recommenders.layouts._helpers import item_placeholder


def dataset_placeholder(root: str | Path) -> str:
    """Prefer a dataset-local placeholder, then the library default."""
    local = Path(root) / "static" / "img" / "poster_not_available.png"
    if local.exists():
        return str(local)
    return item_placeholder()


def resolve_image_urls(items: pd.DataFrame, root: str | Path) -> pd.Series:
    """Resolve item poster paths to usable local paths, falling back to a placeholder.

    Uses ``image_url`` if present, otherwise ``poster_path``; relative paths are
    resolved against ``root`` and missing files fall back to the placeholder.
    """
    placeholder = dataset_placeholder(root)
    root_path = Path(root)

    def resolve(value) -> str:
        if pd.isna(value) or not str(value).strip():
            return placeholder
        path = Path(str(value))
        if not path.is_absolute():
            path = root_path / path
        return str(path) if path.exists() else placeholder

    for column in ("image_url", "poster_path"):
        if column in items.columns:
            return items[column].map(resolve)
    return pd.Series([placeholder] * len(items), index=items.index)
