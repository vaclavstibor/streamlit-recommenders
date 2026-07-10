from streamlit_recommenders.models.adapter import (
    adapt_get_recommendations,
    call_get_recommendations,
)
from streamlit_recommenders.runtime.cache import get_recommendations, hash_params
from streamlit_recommenders.widgets.params import split_model_params


def test_adapt_function():
    def fn(user_id, k, **params):
        return [1, 2, 3][:k]

    adapted = adapt_get_recommendations(fn)
    assert adapted(0, 2) == [1, 2]


def test_adapt_class():
    class Model:
        def get_recommendations(self, user_id, k, **params):
            return [k, k - 1]

    adapted = adapt_get_recommendations(Model())
    assert adapted(0, 2) == [2, 1]


def test_call_get_recommendations():
    def fn(user_id, k, alpha=0.5, session_items=None, selections=None):
        return [user_id] * k + list(session_items or [])

    assert call_get_recommendations(fn, 7, 3, alpha=0.1) == [7, 7, 7]
    assert call_get_recommendations(fn, 7, 3, session_items=[9], alpha=0.1) == [7, 7, 7, 9]
    assert call_get_recommendations(
        fn,
        7,
        3,
        session_items=[9],
        selections=[{"item_id": 9, "rank": 0}],
        alpha=0.1,
    ) == [7, 7, 7, 9]


def test_hash_params_order_independent():
    assert hash_params({"a": 1, "b": 2}) == hash_params({"b": 2, "a": 1})


def test_hash_params_changes_with_values():
    assert hash_params({"a": 1}) != hash_params({"a": 2})


def test_cached_recommendations_overfetch_and_exclude_seen_items():
    requested = []

    def fn(user_id, k, **params):
        requested.append(k)
        return [1, 2, 3, 4, 5][:k]

    result = get_recommendations(fn, 0, 3, {}, excluded_item_ids=[1, 2])

    assert requested == [5]
    assert result == [3, 4, 5]


def test_split_model_params_keeps_global_and_scoped_values():
    global_params, model_params = split_model_params(
        {
            "num_recs": 20,
            "Ours": {"alpha": 0.5},
            "Unknown": {"beta": 1.0},
        },
        ["Ours", "EASE"],
    )

    assert global_params == {"num_recs": 20, "Unknown": {"beta": 1.0}}
    assert model_params == {"Ours": {"alpha": 0.5}}
