from __future__ import annotations

import numpy as np
import pandas as pd

from streamlit_recommenders.runtime.seen import effective_seen, is_session_user


class EmbeddingPopularityRecommender:
    """Blend personal embeddings with popularity; filter items the user already saw."""

    def __init__(
        self,
        user_embeddings: np.ndarray,
        item_embeddings: np.ndarray,
        popularity: np.ndarray,
        interactions: pd.DataFrame,
        items: pd.DataFrame | None = None,
    ) -> None:
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings
        self.popularity = popularity
        self.interactions = interactions
        self.items = items

    @classmethod
    def from_interactions(
        cls,
        user_embeddings: np.ndarray,
        item_embeddings: np.ndarray,
        items: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> EmbeddingPopularityRecommender:
        popularity = (
            interactions.groupby("item_id")
            .size()
            .reindex(range(len(items)), fill_value=0)
            .values
        )
        return cls(user_embeddings, item_embeddings, popularity, interactions, items)

    def _user_vector(
        self,
        user_id: str | int,
        session_items: list | None,
    ) -> np.ndarray:
        session_items = session_items or []
        if session_items:
            session_vec = self.item_embeddings[session_items].mean(axis=0)
            if is_session_user(user_id):
                return session_vec
            base = self.user_embeddings[user_id]
            return 0.6 * session_vec + 0.4 * base
        if is_session_user(user_id):
            return np.zeros(self.item_embeddings.shape[1])
        return self.user_embeddings[user_id]

    def scores(
        self,
        user_id: str | int,
        alpha: float,
        session_items: list | None = None,
    ) -> np.ndarray:
        personal = self._user_vector(user_id, session_items) @ self.item_embeddings.T
        return alpha * personal + (1 - alpha) * self.popularity

    def recommend(
        self,
        user_id: str | int,
        k: int,
        alpha: float = 0.5,
        session_items: list | None = None,
        selections: list[dict] | None = None,
        **params,
    ) -> list[int]:
        session_items = session_items or []
        scores = self.scores(user_id, alpha, session_items)
        seen = effective_seen(self.interactions, user_id, session_items)
        ranked = np.argsort(scores)[::-1]
        return [int(i) for i in ranked if i not in seen][:k]

    def score_frame(
        self,
        user_id: str | int,
        alpha: float,
        session_items: list | None = None,
    ) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "item_id": range(len(self.popularity)),
                "score": self.scores(user_id, alpha, session_items),
            }
        )
        if self.items is not None and "title" in self.items.columns:
            df = df.merge(self.items[["item_id", "title"]], on="item_id")
        return df.sort_values("score", ascending=False)
