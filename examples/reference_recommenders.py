"""Reference recommender definitions.

`streamlit-recommenders` deliberately ships **no** built-in models: the library
owns the interactive layer, and you bring the recommendations. This module is a
worked reference for the second of the three ways to plug in (see the README) --
subclassing :class:`streamlit_recommenders.BaseRecommender` and implementing a
single ``scores`` method. The base class handles seen-filtering and top-k
ranking, so a definition only has to turn a user/session profile into one score
per item.

Three classic collaborative-filtering baselines are implemented here as
copy-and-adapt starting points:

* :class:`ItemKNNRecommender` -- item-item cosine similarity.
* :class:`EASERecommender` -- shallow closed-form linear autoencoder.
* :class:`SequentialCFRecommender` -- next-item transition counts.

Run this file directly to fit all three in memory and compare them:

    SR_DATA_DIR=data/ml-latest-small streamlit run examples/reference_recommenders.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import streamlit_recommenders as sr


def item_ids_from(items: pd.DataFrame | None, interactions: pd.DataFrame) -> list:
    """Resolve the ordered catalog of item ids.

    Args:
        items: Optional items table; its ``item_id`` column defines the catalog
            order when present.
        interactions: Interactions table used as a fallback source of item ids.

    Returns:
        The list of item ids, taken from ``items`` when available and otherwise
        the sorted unique item ids observed in ``interactions``.
    """
    if items is not None and "item_id" in items.columns:
        return items["item_id"].tolist()
    return sorted(interactions["item_id"].unique().tolist())


def user_item_matrix(interactions: pd.DataFrame, item_ids: list) -> tuple[np.ndarray, dict]:
    """Build a dense binary user-item interaction matrix.

    Args:
        interactions: Interactions table with ``user_id`` and ``item_id`` columns.
        item_ids: Ordered catalog of item ids defining the matrix columns.

    Returns:
        A tuple of the ``(n_users, n_items)`` matrix (1.0 where a user interacted
        with an item) and the ``{item_id: column_index}`` lookup.
    """
    user_ids = sorted(interactions["user_id"].unique().tolist())
    user_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
    matrix = np.zeros((len(user_ids), len(item_ids)), dtype=float)
    for row in interactions.itertuples(index=False):
        item_id = getattr(row, "item_id")
        if item_id in item_index:
            matrix[user_index[getattr(row, "user_id")], item_index[item_id]] = 1.0
    return matrix, item_index


def user_vector(
    interactions: pd.DataFrame,
    user_id: str | int,
    item_index: dict,
    session_items: list | None,
) -> np.ndarray:
    """Build a binary profile vector over the catalog for one user.

    Combines the user's historical interactions with the current session's items
    so a fresh "Try yourself" session (no history) is driven purely by clicks.

    Args:
        interactions: Interactions table with ``user_id`` and ``item_id`` columns.
        user_id: The user whose history to include.
        item_index: ``{item_id: column_index}`` lookup for the catalog.
        session_items: Item ids the user selected this session, if any.

    Returns:
        A length-``len(item_index)`` vector with 1.0 for every item in the
        combined profile.
    """
    vector = np.zeros(len(item_index), dtype=float)
    if user_id in set(interactions["user_id"]):
        for item_id in interactions.loc[interactions.user_id == user_id, "item_id"]:
            if item_id in item_index:
                vector[item_index[item_id]] = 1.0
    for item_id in session_items or []:
        if item_id in item_index:
            vector[item_index[item_id]] = 1.0
    return vector


class ItemKNNRecommender(sr.BaseRecommender):
    """Item-item collaborative filtering baseline using cosine similarity.

    Scores are the profile vector projected through a top-``k`` truncated
    item-item cosine similarity matrix.

    Attributes:
        interactions: Interactions table the model was fit on.
        item_ids: Ordered catalog of item ids.
        item_index: ``{item_id: index}`` lookup into ``item_ids``.
        popularity: Per-item interaction counts, used as a cold-start fallback.
        similarity: The fitted ``(n_items, n_items)`` similarity matrix.
    """

    def __init__(
        self,
        interactions: pd.DataFrame,
        items: pd.DataFrame | None = None,
        k_neighbors: int = 50,
    ) -> None:
        """Fit the similarity matrix.

        Args:
            interactions: Interactions table with ``user_id`` and ``item_id``.
            items: Optional items table defining the catalog order.
            k_neighbors: Number of neighbors to keep per item; ``0`` keeps all.
        """
        self.interactions = interactions
        self.item_ids = item_ids_from(items, interactions)
        self.k_neighbors = k_neighbors
        matrix, self.item_index = user_item_matrix(interactions, self.item_ids)
        self.popularity = matrix.sum(axis=0)
        self.similarity = self._fit_similarity(matrix)

    @classmethod
    def from_interactions(
        cls,
        interactions: pd.DataFrame,
        items: pd.DataFrame | None = None,
        k_neighbors: int = 50,
    ) -> ItemKNNRecommender:
        """Fit a model from an interactions table (the in-memory train step).

        Args:
            interactions: Interactions table with ``user_id`` and ``item_id``.
            items: Optional items table defining the catalog order.
            k_neighbors: Number of neighbors to keep per item.

        Returns:
            A fitted :class:`ItemKNNRecommender`.
        """
        return cls(interactions, items, k_neighbors)

    def scores(
        self,
        user_id: str | int,
        session_items: list | None = None,
        **params,
    ) -> np.ndarray:
        """Score every item for a user/session profile.

        Args:
            user_id: The active user (or the session user).
            session_items: Item ids selected this session.
            **params: Unused sidebar params, accepted for contract compatibility.

        Returns:
            One score per item in ``self.item_ids``; popularity is returned for
            an empty profile.
        """
        vector = user_vector(self.interactions, user_id, self.item_index, session_items)
        return vector @ self.similarity if vector.any() else self.popularity.copy()

    def _fit_similarity(self, matrix: np.ndarray) -> np.ndarray:
        """Compute the top-``k`` truncated cosine similarity matrix."""
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
        if 0 < self.k_neighbors < similarity.shape[1]:
            for row in range(similarity.shape[0]):
                keep = np.argpartition(similarity[row], -self.k_neighbors)[-self.k_neighbors :]
                mask = np.ones(similarity.shape[1], dtype=bool)
                mask[keep] = False
                similarity[row, mask] = 0.0
        return similarity


class EASERecommender(sr.BaseRecommender):
    """Embarrassingly Shallow Autoencoder (EASE) baseline for implicit feedback.

    Fits a closed-form item-item weight matrix; scores are the profile vector
    projected through those weights.

    Attributes:
        interactions: Interactions table the model was fit on.
        item_ids: Ordered catalog of item ids.
        item_index: ``{item_id: index}`` lookup into ``item_ids``.
        popularity: Per-item interaction counts, used as a cold-start fallback.
        weights: The fitted ``(n_items, n_items)`` weight matrix.
    """

    def __init__(
        self,
        interactions: pd.DataFrame,
        items: pd.DataFrame | None = None,
        l2: float = 500.0,
    ) -> None:
        """Fit the weight matrix.

        Args:
            interactions: Interactions table with ``user_id`` and ``item_id``.
            items: Optional items table defining the catalog order.
            l2: L2 regularization strength for the closed-form solve.
        """
        self.interactions = interactions
        self.item_ids = item_ids_from(items, interactions)
        matrix, self.item_index = user_item_matrix(interactions, self.item_ids)
        self.popularity = matrix.sum(axis=0)
        self.weights = self._fit_weights(matrix, l2)

    @classmethod
    def from_interactions(
        cls,
        interactions: pd.DataFrame,
        items: pd.DataFrame | None = None,
        l2: float = 500.0,
    ) -> EASERecommender:
        """Fit a model from an interactions table (the in-memory train step).

        Args:
            interactions: Interactions table with ``user_id`` and ``item_id``.
            items: Optional items table defining the catalog order.
            l2: L2 regularization strength.

        Returns:
            A fitted :class:`EASERecommender`.
        """
        return cls(interactions, items, l2)

    def scores(
        self,
        user_id: str | int,
        session_items: list | None = None,
        **params,
    ) -> np.ndarray:
        """Score every item for a user/session profile.

        Args:
            user_id: The active user (or the session user).
            session_items: Item ids selected this session.
            **params: Unused sidebar params, accepted for contract compatibility.

        Returns:
            One score per item in ``self.item_ids``; popularity is returned for
            an empty profile.
        """
        vector = user_vector(self.interactions, user_id, self.item_index, session_items)
        return vector @ self.weights if vector.any() else self.popularity.copy()

    @staticmethod
    def _fit_weights(matrix: np.ndarray, l2: float) -> np.ndarray:
        """Solve the closed-form EASE weight matrix with a zeroed diagonal."""
        gram = matrix.T @ matrix
        diagonal = np.diag_indices(gram.shape[0])
        gram[diagonal] += l2
        precision = np.linalg.pinv(gram)
        weights = -precision / np.diag(precision)
        weights[diagonal] = 0.0
        return weights


class SequentialCFRecommender(sr.BaseRecommender):
    """Lightweight next-item baseline from item-to-item transition counts.

    Scores candidates from the last item in the profile using normalized
    first-order transition probabilities.

    Attributes:
        interactions: Interactions table the model was fit on.
        item_ids: Ordered catalog of item ids.
        item_index: ``{item_id: index}`` lookup into ``item_ids``.
        transitions: Row-normalized ``(n_items, n_items)`` transition matrix.
        popularity: Per-item interaction counts, used as a cold-start fallback.
    """

    def __init__(
        self,
        interactions: pd.DataFrame,
        items: pd.DataFrame | None = None,
        timestamp_col: str = "timestamp",
    ) -> None:
        """Fit the transition matrix.

        Args:
            interactions: Interactions table with ``user_id`` and ``item_id`` and,
                ideally, ``timestamp`` to order each user's sequence.
            items: Optional items table defining the catalog order.
            timestamp_col: Column used to order each user's interactions.
        """
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
        """Fit a model from an interactions table (the in-memory train step).

        Args:
            interactions: Interactions table with ``user_id`` and ``item_id``.
            items: Optional items table defining the catalog order.
            timestamp_col: Column used to order each user's interactions.

        Returns:
            A fitted :class:`SequentialCFRecommender`.
        """
        return cls(interactions, items, timestamp_col)

    def scores(
        self,
        user_id: str | int,
        session_items: list | None = None,
        **params,
    ) -> np.ndarray:
        """Score every item given the last item in the profile.

        Args:
            user_id: The active user (or the session user).
            session_items: Item ids selected this session; the last one is the anchor.
            **params: Unused sidebar params, accepted for contract compatibility.

        Returns:
            One score per item in ``self.item_ids``; popularity is returned when
            the anchor is unknown or has no outgoing transitions.
        """
        anchor = self._last_item(user_id, session_items)
        scores = (
            self.transitions[self.item_index[anchor]].copy()
            if anchor in self.item_index
            else self.popularity.copy()
        )
        if not scores.any():
            scores = self.popularity.copy()
        return scores

    def _fit_transitions(self, interactions: pd.DataFrame) -> np.ndarray:
        """Count and row-normalize first-order item transitions per user."""
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
        """Return the most recent item in the session or the user's history."""
        if session_items:
            return session_items[-1]
        user_rows = self.interactions.loc[self.interactions.user_id == user_id]
        if user_rows.empty:
            return None
        if self.timestamp_col in user_rows.columns:
            user_rows = user_rows.sort_values(self.timestamp_col)
        return user_rows.iloc[-1]["item_id"]


if __name__ == "__main__":
    import os

    DATA_DIR = os.environ.get("SR_DATA_DIR", "data/ml-latest-small")
    ITEMS, TRAIN, _ = sr.load_local_dataset(DATA_DIR)
    MODELS = {
        "ItemKNN": ItemKNNRecommender.from_interactions(TRAIN, ITEMS),
        "EASE": EASERecommender.from_interactions(TRAIN, ITEMS),
        "Sequential CF": SequentialCFRecommender.from_interactions(TRAIN, ITEMS),
    }

    def intro() -> None:
        sr.markdown(
            "### Define Your Own Recommenders\n"
            "The library ships no built-in models. These three baselines are defined in "
            "`examples/reference_recommenders.py` by subclassing `sr.BaseRecommender` and "
            "implementing a single `scores()` method — copy one as a starting point for your "
            "own model. Here they are fit in memory with `from_interactions(train, items)` and "
            "compared on the same session profile."
        )

    sr.run(
        get_recommendations=MODELS,
        items=ITEMS,
        interactions=TRAIN,
        params={"num_recs": sr.selectbox("Number of recommendations", [10, 20, 30], index=1)},
        title="Reference Recommenders",
        subtitle="Define models by subclassing BaseRecommender, then compare them in stacked rows.",
        intro=intro,
    )
