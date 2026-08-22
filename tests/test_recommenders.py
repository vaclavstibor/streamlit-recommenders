import sys
from pathlib import Path

import numpy as np
import pandas as pd

from streamlit_recommenders.recommenders import ArtifactRecommender

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from reference_recommenders import (
    EASERecommender,
    ItemKNNRecommender,
    SequentialCFRecommender,
)


def test_item_knn_recommender_filters_seen():
    interactions = pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1],
            "item_id": [0, 1, 1, 2],
        }
    )
    items = pd.DataFrame({"item_id": [0, 1, 2, 3]})
    model = ItemKNNRecommender.from_interactions(interactions, items, k_neighbors=2)

    recs = model.get_recommendations(0, 2)

    assert 0 not in recs
    assert 1 not in recs
    assert len(recs) <= 2


def test_ease_recommender_filters_session_items():
    interactions = pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1],
            "item_id": [0, 1, 1, 2],
        }
    )
    items = pd.DataFrame({"item_id": [0, 1, 2, 3]})
    model = EASERecommender.from_interactions(interactions, items, l2=10.0)

    recs = model.get_recommendations(0, 2, session_items=[2])

    assert 0 not in recs
    assert 1 not in recs
    assert 2 not in recs


def test_sequential_cf_recommender_uses_last_item():
    interactions = pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1],
            "item_id": [0, 1, 1, 2],
            "timestamp": [1, 2, 1, 2],
        }
    )
    items = pd.DataFrame({"item_id": [0, 1, 2, 3]})
    model = SequentialCFRecommender.from_interactions(interactions, items)

    assert model.get_recommendations(0, 1) == [2]


def test_artifact_fallback_reason(tmp_path):
    path = tmp_path / "m.npz"
    np.savez(
        path,
        model_type=np.array(["itemknn"]),
        item_ids=np.array([1, 2, 3]),
        weights=np.eye(3, dtype=np.float32),
        popularity=np.array([5, 3, 1], dtype=np.float32),
    )
    interactions = pd.DataFrame({"user_id": [7], "item_id": [2]})
    model = ArtifactRecommender(str(path), interactions)

    # A user with mappable history scores through the model.
    assert model.fallback_reason(7) is None
    # The empty-profile session user has no signal -> popularity fallback.
    assert model.fallback_reason("__session__") == "popularity"
    # Once the session has an item, the model has signal again.
    assert model.fallback_reason("__session__", session_items=[1]) is None
