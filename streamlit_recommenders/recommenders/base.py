from __future__ import annotations

import numpy as np
import pandas as pd

from streamlit_recommenders.recommenders._common import rank_scores
from streamlit_recommenders.runtime.seen import effective_seen


class BaseRecommender:
    """Shared contract for score-based recommenders.

    Subclasses provide ``scores()`` returning one value per ``self.item_ids``;
    the base ranks unseen items via the common seen-filtering + top-k logic.
    """

    interactions: pd.DataFrame | None
    item_ids: list

    def scores(
        self,
        user_id: str | int,
        session_items: list | None = None,
        **params,
    ) -> np.ndarray:
        raise NotImplementedError

    def get_recommendations(
        self,
        user_id: str | int,
        k: int,
        session_items: list | None = None,
        selections: list[dict] | None = None,
        **params,
    ) -> list:
        # ``selections`` carries UI feedback metadata; subclasses that want it
        # should override ``get_recommendations``.
        seen = effective_seen(getattr(self, "interactions", None), user_id, session_items)
        scores = self.scores(user_id, session_items=session_items, **params)
        return rank_scores(scores, self.item_ids, seen, k)
