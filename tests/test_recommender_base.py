import sys
from pathlib import Path

import numpy as np

from streamlit_recommenders.recommenders import ArtifactRecommender, BaseRecommender

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from reference_recommenders import (
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


def test_base_get_recommendations_hide_seen_toggle():
    import pandas as pd

    class _Stub(BaseRecommender):
        interactions = pd.DataFrame({"user_id": [0, 0], "item_id": [10, 11]})
        item_ids = [10, 11, 12]

        def scores(self, user_id, session_items=None, **params):
            return np.array([0.9, 0.8, 0.7])

    model = _Stub()

    # Default hides the user's already-seen history items (10, 11).
    assert model.get_recommendations(0, 3) == [12]
    # hide_seen=False exposes the model's raw ranking, seen items included.
    assert model.get_recommendations(0, 3, hide_seen=False) == [10, 11, 12]


DISLIKE = {"item_id": 11, "rank": None, "source": "S", "sentiment": "dislike"}


def test_base_passes_dislikes_only_to_scores_that_declare_them():
    class _Opted(BaseRecommender):
        interactions = None
        item_ids = [10, 11, 12]

        def scores(self, user_id, session_items=None, disliked_items=None, **params):
            self.received = disliked_items
            return np.array([0.1, 0.9, 0.5])

    class _Legacy(BaseRecommender):
        interactions = None
        item_ids = [10, 11, 12]

        def scores(self, user_id, session_items=None, **params):
            assert "disliked_items" not in params
            return np.array([0.1, 0.9, 0.5])

    opted = _Opted()
    # A disliked item counts as seen, whether or not the model reads dislikes.
    assert opted.get_recommendations(0, 3, selections=[DISLIKE]) == [12, 10]
    assert opted.received == [11]
    assert _Legacy().get_recommendations(0, 3, selections=[DISLIKE]) == [12, 10]
