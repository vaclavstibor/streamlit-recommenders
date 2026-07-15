"""Adapt user-supplied recommenders into a uniform callable and call it."""

from collections.abc import Callable
from typing import Any

from streamlit_recommenders.runtime.seen import call_with_session_context, filter_model_params


def adapt_get_recommendations(get_recommendations: Callable[..., list] | Any) -> Callable[..., list]:
    """Normalize a function or object into a get_recommendations callable.

    Args:
        get_recommendations: Either a plain callable, or an object exposing a
            ``get_recommendations`` method.

    Returns:
        A callable with the ``get_recommendations`` signature.

    Raises:
        TypeError: If the input is neither callable nor has a callable
            ``get_recommendations`` attribute.
    """
    if hasattr(get_recommendations, "get_recommendations") and callable(get_recommendations.get_recommendations):
        return get_recommendations.get_recommendations  # type: ignore[union-attr]

    if callable(get_recommendations):
        return get_recommendations

    raise TypeError(
        "get_recommendations must be a function(user_id, k, **params) "
        "or an object with .get_recommendations()"
    )


def call_get_recommendations(
    get_recommendations: Callable[..., list],
    user_id: str | int,
    k: int,
    session_items: list | None = None,
    selections: list[dict] | None = None,
    **params: Any,
) -> list:
    """Invoke a recommender, passing session context only when it is present.

    When session items or selections exist, the call is routed through
    ``call_with_session_context`` so those are forwarded; otherwise the
    recommender is called with just its supported model parameters.

    Args:
        get_recommendations: The recommender callable to invoke.
        user_id: Id of the user to recommend for.
        k: Number of item ids to request.
        session_items: Items selected during the current session.
        selections: Optional UI feedback metadata.
        **params: Model-specific parameters.

    Returns:
        The recommended item ids returned by the recommender.
    """
    if session_items or selections:
        return call_with_session_context(
            get_recommendations,
            user_id,
            k,
            list(session_items or []),
            selections,
            **params,
        )
    return get_recommendations(user_id, k, **filter_model_params(get_recommendations, params))
