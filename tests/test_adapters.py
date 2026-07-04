from streamlit_recommenders.models.adapter import adapt_recommender, call_recommend
from streamlit_recommenders.runtime.cache import hash_params


def test_adapt_function():
    def fn(user_id, k, **params):
        return [1, 2, 3][:k]

    adapted = adapt_recommender(fn)
    assert adapted(0, 2) == [1, 2]


def test_adapt_class():
    class Model:
        def recommend(self, user_id, k, **params):
            return [k, k - 1]

    adapted = adapt_recommender(Model())
    assert adapted(0, 2) == [2, 1]


def test_call_recommend():
    def fn(user_id, k, alpha=0.5, session_items=None, selections=None):
        return [user_id] * k + list(session_items or [])

    assert call_recommend(fn, 7, 3, alpha=0.1) == [7, 7, 7]
    assert call_recommend(fn, 7, 3, session_items=[9], alpha=0.1) == [7, 7, 7, 9]
    assert call_recommend(
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
