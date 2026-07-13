from __future__ import annotations

from pathlib import Path

import pandas as pd

_PLACEHOLDER_URL = "https://placehold.co/130x195/e5e7eb/6b7280?text=Item"
_PLACEHOLDER_FILE = (
    Path(__file__).resolve().parent.parent / "assets" / "poster_not_available.png"
)


def item_placeholder() -> str:
    """Path to the bundled placeholder poster, or a hosted fallback URL."""
    return str(_PLACEHOLDER_FILE) if _PLACEHOLDER_FILE.exists() else _PLACEHOLDER_URL


def dataset_placeholder(root: str | Path) -> str:
    """Prefer a dataset-local placeholder, then the library default."""
    local = Path(root) / "static" / "img" / "poster_not_available.png"
    if local.exists():
        return str(local)
    return item_placeholder()


def resolve_image_urls(items: pd.DataFrame, root: str | Path) -> pd.Series:
    """Resolve item poster references to usable paths or URLs.

    Checks ``image_url`` then ``poster_path`` per item: remote URLs pass
    through unchanged, relative paths are resolved against ``root``, and
    items whose references are missing or unresolvable fall back to the
    placeholder.
    """
    placeholder = dataset_placeholder(root)
    root_path = Path(root)
    columns = [column for column in ("image_url", "poster_path") if column in items.columns]
    if not columns:
        return pd.Series([placeholder] * len(items), index=items.index)

    def resolve_value(value) -> str | None:
        if pd.isna(value) or not str(value).strip():
            return None
        text = str(value)
        if text.startswith(("http://", "https://")):
            return text
        path = Path(text)
        if not path.is_absolute():
            path = root_path / path
        return str(path) if path.exists() else None

    def resolve_row(row: pd.Series) -> str:
        for column in columns:
            resolved = resolve_value(row[column])
            if resolved is not None:
                return resolved
        return placeholder

    return items[columns].apply(resolve_row, axis=1)
