from __future__ import annotations

import numpy as np
import pandas as pd

from streamlit_recommenders.recommenders._common import (
    item_ids_from,
    rank_scores,
    user_item_matrix,
    user_vector,
)
from streamlit_recommenders.runtime.seen import effective_seen


class EASERecommender:
    """Embarrassingly Shallow Autoencoder baseline for implicit feedback."""

    def __init__(
        self,
        interactions: pd.DataFrame,
        items: pd.DataFrame | None = None,
        l2: float = 500.0,
    ) -> None:
        self.interactions = interactions
        self.item_ids = item_ids_from(items, interactions)
        matrix, _, self.item_index = user_item_matrix(interactions, self.item_ids)
        self.popularity = matrix.sum(axis=0)
        self.weights = self._fit_weights(matrix, l2)

    @classmethod
    def from_interactions(
        cls,
        interactions: pd.DataFrame,
        items: pd.DataFrame | None = None,
        l2: float = 500.0,
    ) -> EASERecommender:
        return cls(interactions, items, l2)

    def get_recommendations(
        self,
        user_id: str | int,
        k: int,
        session_items: list | None = None,
        **params,
    ) -> list:
        seen = effective_seen(self.interactions, user_id, session_items)
        vector = user_vector(self.interactions, user_id, self.item_index, session_items)
        scores = vector @ self.weights if vector.any() else self.popularity.copy()
        return rank_scores(scores, self.item_ids, seen, k)

    def score_frame(
        self,
        user_id: str | int,
        session_items: list | None = None,
    ) -> pd.DataFrame:
        vector = user_vector(self.interactions, user_id, self.item_index, session_items)
        scores = vector @ self.weights if vector.any() else self.popularity.copy()
        return pd.DataFrame({"item_id": self.item_ids, "score": scores}).sort_values(
            "score",
            ascending=False,
        )

    @staticmethod
    def _fit_weights(matrix: np.ndarray, l2: float) -> np.ndarray:
        gram = matrix.T @ matrix
        diagonal = np.diag_indices(gram.shape[0])
        gram[diagonal] += l2
        precision = np.linalg.pinv(gram)
        weights = -precision / np.diag(precision)
        weights[diagonal] = 0.0
        return weights


ELSARecommender = EASERecommender
