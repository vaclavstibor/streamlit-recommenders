from __future__ import annotations

import numpy as np
import pandas as pd

from streamlit_recommenders.runtime.seen import effective_seen


class RandomRecommender:
    """Sample unseen items uniformly at random (deterministic per user for caching)."""

    def __init__(
        self,
        interactions: pd.DataFrame,
        n_items: int | None = None,
        item_ids: list | None = None,
        seed: int = 0,
    ) -> None:
        self.interactions = interactions
        self.item_ids = item_ids or list(range(n_items or 0))
        self.seed = seed

    def get_recommendations(
        self,
        user_id: str | int,
        k: int,
        session_items: list | None = None,
        **params,
    ) -> list:
        seen = effective_seen(self.interactions, user_id, session_items)
        candidates = [item_id for item_id in self.item_ids if item_id not in seen]
        if not candidates:
            return []
        user_seed = self.seed + (
            int(user_id) if isinstance(user_id, int) else hash(user_id) % 2**32
        )
        rng = np.random.default_rng(user_seed + len(session_items or []))
        size = min(k, len(candidates))
        choice = rng.choice(candidates, size=size, replace=False)
        return [int(i) for i in choice]
