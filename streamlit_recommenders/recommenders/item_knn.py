from __future__ import annotations

import numpy as np
import pandas as pd

from streamlit_recommenders.recommenders._common import (
    item_ids_from,
    user_item_matrix,
    user_vector,
)
from streamlit_recommenders.recommenders.base import BaseRecommender


class ItemKNNRecommender(BaseRecommender):
    """Item-item collaborative filtering baseline using cosine similarity."""

    def __init__(
        self,
        interactions: pd.DataFrame,
        items: pd.DataFrame | None = None,
        k_neighbors: int = 50,
    ) -> None:
        self.interactions = interactions
        self.item_ids = item_ids_from(items, interactions)
        self.k_neighbors = k_neighbors
        matrix, _, self.item_index = user_item_matrix(interactions, self.item_ids)
        self.popularity = matrix.sum(axis=0)
        self.similarity = self._fit_similarity(matrix)

    @classmethod
    def from_interactions(
        cls,
        interactions: pd.DataFrame,
        items: pd.DataFrame | None = None,
        k_neighbors: int = 50,
    ) -> ItemKNNRecommender:
        return cls(interactions, items, k_neighbors)

    def scores(
        self,
        user_id: str | int,
        session_items: list | None = None,
        **params,
    ) -> np.ndarray:
        vector = user_vector(self.interactions, user_id, self.item_index, session_items)
        return vector @ self.similarity if vector.any() else self.popularity.copy()

    def _fit_similarity(self, matrix: np.ndarray) -> np.ndarray:
        cooccurrence = matrix.T @ matrix
        norms = np.sqrt(np.diag(cooccurrence))
        denom = np.outer(norms, norms)
        similarity = np.divide(
            cooccurrence,
            denom,
            out=np.zeros_like(cooccurrence, dtype=float),
            where=denom > 0,
        )
        np.fill_diagonal(similarity, 0.0)
        if self.k_neighbors > 0 and self.k_neighbors < similarity.shape[1]:
            for row in range(similarity.shape[0]):
                keep = np.argpartition(similarity[row], -self.k_neighbors)[-self.k_neighbors :]
                mask = np.ones(similarity.shape[1], dtype=bool)
                mask[keep] = False
                similarity[row, mask] = 0.0
        return similarity
