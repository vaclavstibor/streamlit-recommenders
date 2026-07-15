"""Structural protocol that user-supplied recommenders must satisfy."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RecommenderProtocol(Protocol):
    """Contract a recommender object implements to plug into the demo.

    Any object exposing a matching ``get_recommendations`` method is accepted;
    ``runtime_checkable`` allows ``isinstance`` checks against this protocol.
    """

    def get_recommendations(
        self,
        user_id: str | int,
        k: int,
        session_items: list[str | int] | None = None,
        selections: list[dict] | None = None,
        **params: Any,
    ) -> list[str | int]:
        """Return the top-``k`` recommended item ids for a user.

        Args:
            user_id: Id of the user to recommend for.
            k: Number of item ids to return.
            session_items: Items selected during the current session.
            selections: Optional UI feedback metadata (e.g. likes/dislikes).
            **params: Model-specific parameters supplied by the sidebar.

        Returns:
            Up to ``k`` recommended item ids ordered by relevance.
        """
        ...
