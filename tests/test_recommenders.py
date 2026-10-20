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


def _artifact(tmp_path, model_type, weights):
    path = tmp_path / f"{model_type}.npz"
    n = weights.shape[0]
    np.savez(
        path,
        model_type=np.array([model_type]),
        item_ids=np.arange(1, n + 1),
        weights=weights.astype(np.float32),
        popularity=np.arange(n, 0, -1).astype(np.float32),
    )
    return ArtifactRecommender(str(path), pd.DataFrame({"user_id": [], "item_id": []}))


def _dislike(item_id):
    return {"item_id": item_id, "rank": None, "source": "S", "sentiment": "dislike"}


def test_artifact_dislike_demotes_look_alikes(tmp_path):
    # Liking 1 favours 2 over 3; item 4 is a near-duplicate of 2.
    weights = np.array(
        [
            [0.0, 0.9, 0.6, 0.0],
            [0.9, 0.0, 0.0, 0.0],
            [0.6, 0.0, 0.0, 0.0],
            [0.0, 0.8, 0.0, 0.0],
        ]
    )
    model = _artifact(tmp_path, "ease", weights)

    assert model.get_recommendations("__session__", 2, session_items=[1]) == [2, 3]
    # Disliking 4 (which resembles 2) pushes 2 below 3.
    recs = model.get_recommendations("__session__", 2, session_items=[1], selections=[_dislike(4)])
    assert recs == [3, 2]
    # dislike_weight=0 turns the penalty off; 4 is still excluded as seen.
    recs = model.get_recommendations(
        "__session__", 3, session_items=[1], selections=[_dislike(4)], dislike_weight=0.0
    )
    assert recs == [2, 3]


def test_artifact_dislikes_alone_keep_the_popularity_fallback(tmp_path):
    model = _artifact(tmp_path, "ease", np.ones((3, 3)) - np.eye(3))

    recs = model.get_recommendations("__session__", 2, selections=[_dislike(1)])

    assert recs == [2, 3]  # popularity order, disliked item excluded
    assert model.fallback_reason("__session__") == "popularity"


def test_sequential_artifact_dislike_demotes_typical_successors(tmp_path):
    # From anchor 1 the usual next items are 2 then 3; item 4 is usually followed by 2.
    weights = np.zeros((4, 4))
    weights[0, 1], weights[0, 2] = 0.6, 0.4
    weights[3, 1] = 1.0
    model = _artifact(tmp_path, "sequential_cf", weights)

    assert model.get_recommendations("__session__", 1, session_items=[1]) == [2]
    recs = model.get_recommendations("__session__", 1, session_items=[1], selections=[_dislike(4)])
    assert recs == [3]
