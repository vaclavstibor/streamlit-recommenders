from collections.abc import Callable
from typing import Any

from streamlit_recommenders.runtime.seen import call_with_session_context, filter_model_params


def adapt_recommender(recommend: Callable[..., list] | Any) -> Callable[..., list]:
    """Normalize a function or object with .recommend() into a single callable."""
    if callable(recommend) and not hasattr(recommend, "recommend"):
        return recommend

    if hasattr(recommend, "recommend") and callable(recommend.recommend):
        return recommend.recommend  # type: ignore[union-attr]

    raise TypeError(
        "recommend must be a function(user_id, k, **params) or an object with .recommend()"
    )


def call_recommend(
    recommend: Callable[..., list],
    user_id: str | int,
    k: int,
    session_items: list | None = None,
    selections: list[dict] | None = None,
    **params: Any,
) -> list:
    if session_items or selections:
        return call_with_session_context(
            recommend,
            user_id,
            k,
            list(session_items or []),
            selections,
            **params,
        )
    return recommend(user_id, k, **filter_model_params(recommend, params))
