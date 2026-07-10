from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from streamlit_recommenders.recommenders.base import BaseRecommender


class ArtifactRecommender(BaseRecommender):
    """Expose exported NumPy artifacts (item_ids, weights, popularity) through the contract."""

    def __init__(
        self,
        artifact_path: str | Path,
        interactions: pd.DataFrame,
    ) -> None:
        artifact = np.load(artifact_path, allow_pickle=False)
        self.name = Path(artifact_path).stem
        self.model_type = str(artifact["model_type"][0])
        self.item_ids = artifact["item_ids"].tolist()
        self.item_index = {item_id: idx for idx, item_id in enumerate(self.item_ids)}
        self.weights = artifact["weights"]
        self.popularity = artifact["popularity"]
        self.interactions = interactions
        self.user_history = self._build_user_history(interactions)

    def scores(
        self,
        user_id: str | int,
        session_items: list | None = None,
        history_window: int | str | None = None,
        **params,
    ) -> np.ndarray:
        session_items = session_items or []
        if self.model_type == "sequential_cf":
            anchor = self._last_item(user_id, session_items)
            if anchor in self.item_index:
                scores = self.weights[self.item_index[anchor]].copy()
                if scores.any():
                    return scores
            return self.popularity.copy()

        vector = self._user_vector(user_id, session_items, history_window=history_window)
        if not vector.any():
            return self.popularity.copy()
        return vector @ self.weights

    def score_frame(
        self,
        user_id: str | int,
        session_items: list | None = None,
        items: pd.DataFrame | None = None,
        **params,
    ) -> pd.DataFrame:
        scores = self.scores(user_id, session_items=session_items, **params)
        frame = pd.DataFrame({"item_id": self.item_ids, "score": scores})
        if items is not None and {"item_id", "title"}.issubset(items.columns):
            frame = frame.merge(items[["item_id", "title"]], on="item_id", how="left")
        return frame.sort_values("score", ascending=False)

    def _user_vector(
        self,
        user_id: str | int,
        session_items: list,
        *,
        history_window: int | str | None = None,
    ) -> np.ndarray:
        vector = np.zeros(len(self.item_ids), dtype=np.float32)
        history = self.user_history.get(user_id, []) + list(session_items)
        if history_window not in (None, "All"):
            history = history[-int(history_window):]
        for item_id in history:
            if item_id in self.item_index:
                vector[self.item_index[item_id]] = 1.0
        return vector

    def _last_item(self, user_id: str | int, session_items: list) -> str | int | None:
        if session_items:
            return session_items[-1]
        history = self.user_history.get(user_id, [])
        if not history:
            return None
        return history[-1]

    @staticmethod
    def _build_user_history(interactions: pd.DataFrame) -> dict:
        frame = interactions
        if "timestamp" in frame.columns:
            frame = frame.sort_values(["user_id", "timestamp"])
        return frame.groupby("user_id")["item_id"].agg(list).to_dict()
