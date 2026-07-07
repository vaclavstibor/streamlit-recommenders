from __future__ import annotations

import pandas as pd

from streamlit_recommenders.runtime.seen import effective_seen


class PopularityRecommender:
    """Rank unseen items by global interaction count."""

    def __init__(self, interactions: pd.DataFrame) -> None:
        self.interactions = interactions
        self.popularity = interactions.groupby("item_id").size()

    @classmethod
    def from_interactions(cls, interactions: pd.DataFrame) -> PopularityRecommender:
        return cls(interactions)

    def get_recommendations(
        self,
        user_id: str | int,
        k: int,
        session_items: list | None = None,
        **params,
    ) -> list[int]:
        seen = effective_seen(self.interactions, user_id, session_items)
        ranked = self.popularity.sort_values(ascending=False)
        return [int(i) for i in ranked.index if i not in seen][:k]
