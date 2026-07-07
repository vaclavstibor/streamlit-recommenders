from collections.abc import Callable
from typing import Any

from streamlit_recommenders.runtime.seen import call_with_session_context, filter_model_params


def adapt_get_recommendations(get_recommendations: Callable[..., list] | Any) -> Callable[..., list]:
    """Normalize a function or object into a get_recommendations callable."""
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
