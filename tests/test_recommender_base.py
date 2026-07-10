import numpy as np

from streamlit_recommenders.recommenders import (
    ArtifactRecommender,
    BaseRecommender,
    EASERecommender,
    ItemKNNRecommender,
    SequentialCFRecommender,
)


def test_baselines_inherit_base_recommender():
    for cls in (
        ArtifactRecommender,
        EASERecommender,
        ItemKNNRecommender,
        SequentialCFRecommender,
    ):
        assert issubclass(cls, BaseRecommender)


def test_base_get_recommendations_uses_scores_and_filters_seen():
    class _Stub(BaseRecommender):
        interactions = None
        item_ids = [10, 11, 12]

        def scores(self, user_id, session_items=None, **params):
            return np.array([0.1, 0.9, 0.5])

    model = _Stub()

    assert model.get_recommendations(0, 3) == [11, 12, 10]
    assert model.get_recommendations(0, 3, session_items=[11]) == [12, 10]
