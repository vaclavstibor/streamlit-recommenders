import pandas as pd

from streamlit_recommenders.recommenders import (
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
