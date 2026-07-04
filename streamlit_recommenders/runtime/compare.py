from collections.abc import Callable, Mapping
from typing import Any

from streamlit_recommenders.models.adapter import adapt_recommender
from streamlit_recommenders.models.protocol import RecommenderProtocol

RecommendInput = (
    Callable[..., list]
    | RecommenderProtocol
    | Mapping[str, Callable[..., list] | RecommenderProtocol]
)


def labeled_recommenders(recommend: RecommendInput) -> list[tuple[str, Callable[..., list]]]:
    """Normalize a single recommender or a named dict into (label, callable) pairs."""
    if isinstance(recommend, Mapping):
        return [(label, adapt_recommender(model)) for label, model in recommend.items()]
    return [("Recommendations", adapt_recommender(recommend))]


def is_compare_mode(recommend: RecommendInput) -> bool:
    return isinstance(recommend, Mapping)
