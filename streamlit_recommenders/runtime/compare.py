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
    """Normalize a single input or named dict into (label, callable) pairs."""
    if isinstance(get_recommendations, Mapping):
        return [
            (label, adapt_get_recommendations(model))
            for label, model in get_recommendations.items()
        ]
    return [("Recommendations", adapt_get_recommendations(get_recommendations))]


def is_compare_mode(get_recommendations: GetRecommendationsInput) -> bool:
    return isinstance(get_recommendations, Mapping)
