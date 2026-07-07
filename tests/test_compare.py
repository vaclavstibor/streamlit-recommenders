from streamlit_recommenders.models.adapter import adapt_get_recommendations
from streamlit_recommenders.runtime.compare import labeled_get_recommendations


def test_labeled_get_recommendations_single():
    def fn(user_id, k, **params):
        return [1, 2][:k]

    pairs = labeled_get_recommendations(fn)
    assert len(pairs) == 1
    assert pairs[0][0] == "Recommendations"
    assert pairs[0][1](0, 2) == [1, 2]


def test_labeled_get_recommendations_compare():
    def a(user_id, k, **params):
        return [1]

    def b(user_id, k, **params):
        return [2]

    pairs = labeled_get_recommendations({"Model A": a, "Model B": b})
    assert [label for label, _ in pairs] == ["Model A", "Model B"]
    assert pairs[0][1](0, 1) == [1]
    assert pairs[1][1](0, 1) == [2]


def test_labeled_get_recommendations_class():
    class Model:
        def get_recommendations(self, user_id, k, **params):
            return [3]

    pairs = labeled_get_recommendations({"Wrapped": Model()})
    assert adapt_get_recommendations(Model())(0, 1) == [3]
