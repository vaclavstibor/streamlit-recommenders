from typing import Any

import pandas as pd

DEFAULT_COLUMNS = {
    "id": "item_id",
    "title": "title",
    "image": "image_url",
    "description": "description",
}

_PLACEHOLDER = "https://placehold.co/130x195/e5e7eb/6b7280?text=Item"


def item_placeholder() -> str:
    return _PLACEHOLDER


def resolve_columns(columns: dict[str, str] | None) -> dict[str, str]:
    merged = DEFAULT_COLUMNS.copy()
    if columns:
        merged.update(columns)
    return merged


def items_for_recs(
    items: pd.DataFrame,
    rec_ids: list,
    columns: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    cols = resolve_columns(columns)
    id_col = cols["id"]
    indexed = items.set_index(id_col, drop=False)
    result = []
    for item_id in rec_ids:
        if item_id not in indexed.index:
            continue
        row = indexed.loc[item_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        result.append(
            {
                "id": item_id,
                "title": _cell(row, cols["title"], str(item_id)),
                "image": _cell(row, cols["image"], None),
                "description": _cell(row, cols["description"], ""),
            }
        )
    return result


def _cell(row: pd.Series, col: str, default: Any) -> Any:
    if col not in row.index:
        return default
    value = row[col]
    if pd.isna(value):
        return default
    return value
