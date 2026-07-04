from __future__ import annotations

import numpy as np
import pandas as pd


def item_ids_from(items: pd.DataFrame | None, interactions: pd.DataFrame) -> list:
    if items is not None and "item_id" in items.columns:
        return items["item_id"].tolist()
    return sorted(interactions["item_id"].unique().tolist())


def user_item_matrix(
    interactions: pd.DataFrame,
    item_ids: list,
    *,
    value_col: str | None = None,
) -> tuple[np.ndarray, dict, dict]:
    user_ids = sorted(interactions["user_id"].unique().tolist())
    user_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
    matrix = np.zeros((len(user_ids), len(item_ids)), dtype=float)

    for row in interactions.itertuples(index=False):
        user_id = getattr(row, "user_id")
        item_id = getattr(row, "item_id")
        if item_id not in item_index:
            continue
        value = getattr(row, value_col) if value_col and hasattr(row, value_col) else 1.0
        matrix[user_index[user_id], item_index[item_id]] = float(value)

    return matrix, user_index, item_index


def user_vector(
    interactions: pd.DataFrame,
    user_id: str | int,
    item_index: dict,
    session_items: list | None,
) -> np.ndarray:
    vector = np.zeros(len(item_index), dtype=float)
    if user_id in set(interactions["user_id"]):
        for item_id in interactions.loc[interactions.user_id == user_id, "item_id"]:
            if item_id in item_index:
                vector[item_index[item_id]] = 1.0
    for item_id in session_items or []:
        if item_id in item_index:
            vector[item_index[item_id]] = 1.0
    return vector


def rank_scores(
    scores: np.ndarray,
    item_ids: list,
    seen: set,
    k: int,
) -> list:
    ranked = np.argsort(scores)[::-1]
    result = []
    for idx in ranked:
        item_id = item_ids[int(idx)]
        if item_id in seen:
            continue
        result.append(item_id)
        if len(result) >= k:
            break
    return result
