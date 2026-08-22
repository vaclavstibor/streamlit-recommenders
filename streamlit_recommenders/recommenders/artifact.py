"""Recommender backed by pre-trained NumPy artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from streamlit_recommenders.recommenders.base import BaseRecommender


def load_artifacts(
    paths: Mapping[str, str | Path],
    interactions: pd.DataFrame,
) -> dict[str, ArtifactRecommender]:
    """Load a set of exported ``.npz`` artifacts into recommenders.

    The batch companion to :class:`ArtifactRecommender`: pass a mapping of
    display label to artifact path (e.g. ``{"EASE": "artifacts/ease.npz"}``)
    and get back one loaded recommender per label, all sharing the same
    ``interactions`` for user history. The labels are exactly the section
    labels used for side-by-side comparison in :func:`streamlit_recommenders.run`.

    Args:
        paths: Mapping of display label to the artifact's ``.npz`` path.
        interactions: Interactions DataFrame supplying per-user history to
            every loaded recommender.

    Returns:
        A dict mapping each label to its loaded :class:`ArtifactRecommender`.

    Raises:
        FileNotFoundError: If any of the artifact paths does not exist; the
            message lists every missing path.
    """
    missing = [str(path) for path in paths.values() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing artifact(s):\n" + "\n".join(missing))
    return {label: ArtifactRecommender(path, interactions) for label, path in paths.items()}


class ArtifactRecommender(BaseRecommender):
    """Expose exported NumPy artifacts (item_ids, weights, popularity).

    Loads a ``.npz`` artifact and serves recommendations either from an
    item-item weight matrix (item-based CF or sequential CF) or from a
    popularity fallback when the user/session profile is empty.

    Attributes:
        name: Artifact file stem, used as a display name.
        model_type: Artifact model type, e.g. ``"sequential_cf"``.
        item_ids: Ordered item ids aligned with weights and popularity.
        item_index: Mapping from item id to its row index.
        weights: Item-item weight matrix.
        popularity: Global popularity scores used as a fallback.
        interactions: Interactions DataFrame supplying user history.
        user_history: Mapping from user id to their ordered item history.
    """

    def __init__(
        self,
        artifact_path: str | Path,
        interactions: pd.DataFrame,
    ) -> None:
        """Load the artifact and build per-user interaction history.

        Args:
            artifact_path: Path to the exported ``.npz`` artifact.
            interactions: Interactions DataFrame used for user history.
        """
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
        """Score every item for ``user_id`` using the loaded artifact.

        For sequential models, scores come from the weight row of the most
        recent (anchor) item; otherwise from a user profile vector multiplied
        by the weight matrix. Falls back to popularity when no signal exists.

        Args:
            user_id: Id of the user to score items for.
            session_items: Items selected during the current session.
            history_window: Number of most recent history items to keep, or
                ``"All"``/``None`` to use the full history.
            **params: Additional parameters (ignored).

        Returns:
            A score array aligned with ``self.item_ids``.
        """
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

    def fallback_reason(
        self,
        user_id: str | int,
        session_items: list | None = None,
        **params,
    ) -> str | None:
        """Report whether scoring ``user_id`` would fall back to popularity.

        Lets the UI flag a result that is the popularity fallback rather than
        the model's own output, so an empty/unmapped profile (or a sequential
        anchor with no outgoing transitions) is never silently attributed to
        the recommender.

        Args:
            user_id: Id of the user that would be scored.
            session_items: Items selected during the current session.
            **params: Additional parameters; ``history_window`` is honored, the
                rest are ignored.

        Returns:
            ``"popularity"`` if the score call would return the popularity
            fallback, else ``None``.
        """
        session_items = session_items or []
        if self.model_type == "sequential_cf":
            anchor = self._last_item(user_id, session_items)
            if anchor in self.item_index and self.weights[self.item_index[anchor]].any():
                return None
            return "popularity"
        vector = self._user_vector(
            user_id, session_items, history_window=params.get("history_window")
        )
        return None if vector.any() else "popularity"

    def score_frame(
        self,
        user_id: str | int,
        session_items: list | None = None,
        items: pd.DataFrame | None = None,
        **params,
    ) -> pd.DataFrame:
        """Return a score-per-item DataFrame sorted by descending score.

        Args:
            user_id: Id of the user to score items for.
            session_items: Items selected during the current session.
            items: Optional catalog used to attach item titles.
            **params: Additional scoring parameters forwarded to ``scores()``.

        Returns:
            A DataFrame with ``item_id`` and ``score`` columns (plus ``title``
            when ``items`` provides it), sorted by descending score.
        """
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
        """Build a binary profile vector from user history and session items."""
        vector = np.zeros(len(self.item_ids), dtype=np.float32)
        history = self.user_history.get(user_id, []) + list(session_items)
        if history_window not in (None, "All"):
            history = history[-int(history_window):]
        for item_id in history:
            if item_id in self.item_index:
                vector[self.item_index[item_id]] = 1.0
        return vector

    def _last_item(self, user_id: str | int, session_items: list) -> str | int | None:
        """Return the most recent item from session items or user history."""
        if session_items:
            return session_items[-1]
        history = self.user_history.get(user_id, [])
        if not history:
            return None
        return history[-1]

    @staticmethod
    def _build_user_history(interactions: pd.DataFrame) -> dict:
        """Map each user id to their ordered list of interacted item ids."""
        frame = interactions
        if "timestamp" in frame.columns:
            frame = frame.sort_values(["user_id", "timestamp"])
        return frame.groupby("user_id")["item_id"].agg(list).to_dict()
