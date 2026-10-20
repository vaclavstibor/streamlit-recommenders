"""Base class for score-based recommenders."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from streamlit_recommenders.recommenders._common import rank_scores
from streamlit_recommenders.runtime.seen import disliked_item_ids, effective_seen


class BaseRecommender:
    """Shared contract for score-based recommenders.

    Subclasses provide ``scores()`` returning one value per ``self.item_ids``;
    the base ranks unseen items via the common seen-filtering + top-k logic.
    A ``scores()`` that declares a ``disliked_items`` parameter also receives
    the session's swipe dislikes, to use as a negative signal.

    Attributes:
        interactions: Optional interactions DataFrame used to determine which
            items a user has already seen.
        item_ids: Ordered item ids aligned with the score vector.
    """

    interactions: pd.DataFrame | None
    item_ids: list

    def scores(
        self,
        user_id: str | int,
        session_items: list | None = None,
        **params,
    ) -> np.ndarray:
        """Return one score per item in ``self.item_ids`` for ``user_id``.

        Args:
            user_id: Id of the user to score items for.
            session_items: Items selected during the current session.
            **params: Model-specific scoring parameters.

        Returns:
            A score array aligned with ``self.item_ids``.

        Raises:
            NotImplementedError: Always; subclasses must implement scoring.
        """
        raise NotImplementedError

    def get_recommendations(
        self,
        user_id: str | int,
        k: int,
        session_items: list | None = None,
        selections: list[dict] | None = None,
        **params,
    ) -> list:
        """Return the top-``k`` recommended item ids, excluding seen items.

        Args:
            user_id: Id of the user to recommend for.
            k: Number of item ids to return.
            session_items: Items selected during the current session.
            selections: UI feedback metadata. Dislikes in it are treated as
                seen and forwarded to ``scores(disliked_items=...)`` when
                ``scores()`` declares that parameter.
            **params: Model-specific scoring parameters passed to ``scores()``.
                The reserved ``hide_seen`` flag (default ``True``) is consumed
                here rather than forwarded to ``scores()``.

        Returns:
            Up to ``k`` recommended item ids ranked by descending score.
        """
        # ``hide_seen`` lets a demo inspect the model's raw ranking (seen items
        # included) instead of the default already-seen-filtered output.
        hide_seen = params.pop("hide_seen", True)
        disliked = disliked_item_ids(selections)
        seen = (
            effective_seen(getattr(self, "interactions", None), user_id, session_items) | set(disliked)
            if hide_seen
            else set()
        )
        # Opt-in by signature, so existing ``scores(**params)`` overrides never
        # receive an argument they did not ask for.
        if disliked and "disliked_items" in inspect.signature(self.scores).parameters:
            params["disliked_items"] = disliked
        scores = self.scores(user_id, session_items=session_items, **params)
        return rank_scores(scores, self.item_ids, seen, k)
