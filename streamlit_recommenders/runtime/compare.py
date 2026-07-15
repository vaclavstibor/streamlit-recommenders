"""Normalization of recommender inputs into labeled compare-mode pairs."""

from collections.abc import Callable, Mapping
from typing import Any

from streamlit_recommenders.models.adapter import adapt_get_recommendations
from streamlit_recommenders.models.protocol import RecommenderProtocol

GetRecommendationsInput = (
    Callable[..., list]
    | RecommenderProtocol
    | Mapping[str, Callable[..., list] | RecommenderProtocol]
)


def labeled_get_recommendations(
    get_recommendations: GetRecommendationsInput,
) -> list[tuple[str, Callable[..., list]]]:
    """Normalize a single input or named dict into (label, callable) pairs.

    A mapping yields one pair per entry (compare mode); a single model or
    callable yields one pair labeled ``"Recommendations"``. Each model is run
    through :func:`adapt_get_recommendations` to a uniform callable.

    Args:
        get_recommendations: A callable, a recommender, or a name-to-model map.

    Returns:
        A list of ``(label, callable)`` pairs.
    """
    if isinstance(get_recommendations, Mapping):
        return [
            (label, adapt_get_recommendations(model))
            for label, model in get_recommendations.items()
        ]
    return [("Recommendations", adapt_get_recommendations(get_recommendations))]


def is_compare_mode(get_recommendations: GetRecommendationsInput) -> bool:
    """Return whether the input asks for side-by-side model comparison.

    Args:
        get_recommendations: The recommender input to inspect.

    Returns:
        ``True`` when a mapping of named models was supplied, else ``False``.
    """
    return isinstance(get_recommendations, Mapping)
