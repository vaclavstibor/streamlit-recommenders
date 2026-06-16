from collections.abc import Callable
from typing import Any

from streamlit_recommenders.models.protocol import RecommenderProtocol


def adapt_recommender(recommend: Callable[..., list] | RecommenderProtocol) -> Callable[..., list]:
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
    **params: Any,
) -> list:
    return recommend(user_id, k, **params)
