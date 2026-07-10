from __future__ import annotations

import numpy as np
import pandas as pd

from streamlit_recommenders.recommenders._common import item_ids_from
from streamlit_recommenders.recommenders.base import BaseRecommender


class SequentialCFRecommender(BaseRecommender):
    """Lightweight next-item baseline from item-to-item transition counts."""

    def __init__(
        self,
        interactions: pd.DataFrame,
        items: pd.DataFrame | None = None,
        timestamp_col: str = "timestamp",
    ) -> None:
        self.interactions = interactions
        self.item_ids = item_ids_from(items, interactions)
        self.item_index = {item_id: idx for idx, item_id in enumerate(self.item_ids)}
        self.timestamp_col = timestamp_col
        self.transitions = self._fit_transitions(interactions)
        self.popularity = (
            interactions.groupby("item_id").size().reindex(self.item_ids, fill_value=0).to_numpy()
        )

    @classmethod
    def from_interactions(
        cls,
        interactions: pd.DataFrame,
        items: pd.DataFrame | None = None,
        timestamp_col: str = "timestamp",
    ) -> SequentialCFRecommender:
        return cls(interactions, items, timestamp_col)

    def scores(
        self,
        user_id: str | int,
        session_items: list | None = None,
        **params,
    ) -> np.ndarray:
        anchor = self._last_item(user_id, session_items)
        scores = self.transitions[self.item_index[anchor]].copy() if anchor in self.item_index else self.popularity.copy()
        if not scores.any():
            scores = self.popularity.copy()
        return scores

    def _fit_transitions(self, interactions: pd.DataFrame) -> np.ndarray:
        transitions = np.zeros((len(self.item_ids), len(self.item_ids)), dtype=float)
        for _, group in interactions.groupby("user_id"):
            if self.timestamp_col in group.columns:
                group = group.sort_values(self.timestamp_col)
            sequence = [item for item in group["item_id"].tolist() if item in self.item_index]
            for current, nxt in zip(sequence, sequence[1:]):
                transitions[self.item_index[current], self.item_index[nxt]] += 1.0
        row_sums = transitions.sum(axis=1, keepdims=True)
        return np.divide(
            transitions,
            row_sums,
            out=np.zeros_like(transitions),
            where=row_sums > 0,
        )

    def _last_item(self, user_id: str | int, session_items: list | None) -> str | int | None:
        if session_items:
            return session_items[-1]
        user_rows = self.interactions.loc[self.interactions.user_id == user_id]
        if user_rows.empty:
            return None
        if self.timestamp_col in user_rows.columns:
            user_rows = user_rows.sort_values(self.timestamp_col)
        return user_rows.iloc[-1]["item_id"]
