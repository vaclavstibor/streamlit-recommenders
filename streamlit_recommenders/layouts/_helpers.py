from typing import Any

import pandas as pd

from streamlit_recommenders.data import ColumnMap
from streamlit_recommenders.data.media import item_placeholder

DEFAULT_COLUMNS = ColumnMap().item_columns()


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
                "image": _cell(row, cols["image"], item_placeholder()),
                "description": _cell(row, cols["description"], ""),
            }
        )
    return result


def visible_entries(
    items: pd.DataFrame,
    rec_ids: list,
    selected_ids: set[str | int],
    columns: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    visible_ids: list = []
    seen: set[str | int] = set()
    for item_id in rec_ids:
        if item_id in seen:
            continue
        seen.add(item_id)
        visible_ids.append(item_id)
    entries = items_for_recs(items, visible_ids, columns)
    for entry in entries:
        entry["selected"] = entry["id"] in selected_ids
    return entries


def _cell(row: pd.Series, col: str, default: Any) -> Any:
    if col not in row.index:
        return default
    value = row[col]
    if pd.isna(value):
        return default
    if isinstance(value, str) and not value.strip():
        return default
    return value
