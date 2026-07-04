import numpy as np
import pandas as pd

from streamlit_recommenders.models.adapter import adapt_recommender
from streamlit_recommenders.recommenders import (
    EASERecommender,
    EmbeddingPopularityRecommender,
    ItemKNNRecommender,
    PopularityRecommender,
    RandomRecommender,
    SequentialCFRecommender,
)
from streamlit_recommenders.runtime.seen import SESSION_USER_ID


def test_embedding_popularity_recommender():
    items = pd.DataFrame({"item_id": [0, 1, 2], "title": ["a", "b", "c"]})
    interactions = pd.DataFrame({"user_id": [0, 0], "item_id": [0, 1]})
    user_emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    item_emb = np.eye(3, 2)

    model = EmbeddingPopularityRecommender.from_interactions(user_emb, item_emb, items, interactions)
    adapted = adapt_recommender(model)

    assert adapted(0, 2, alpha=1.0) == [2]
    assert adapted(1, 2, alpha=1.0) == [1, 2]


def test_popularity_recommender():
    interactions = pd.DataFrame({"user_id": [0, 0, 1], "item_id": [0, 0, 1]})
    model = PopularityRecommender.from_interactions(interactions)
    assert model.recommend(0, 2) == [1]
    assert model.recommend(1, 2) == [0]


def test_embedding_popularity_session_user():
    items = pd.DataFrame({"item_id": [0, 1, 2], "title": ["a", "b", "c"]})
    interactions = pd.DataFrame({"user_id": [0, 0], "item_id": [0, 1]})
    user_emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    item_emb = np.eye(3, 2)

    model = EmbeddingPopularityRecommender.from_interactions(user_emb, item_emb, items, interactions)
    recs = model.recommend(SESSION_USER_ID, 2, alpha=1.0, session_items=[0])
    assert 0 not in recs
    assert len(recs) <= 2


def test_random_recommender_deterministic():
    interactions = pd.DataFrame({"user_id": [0], "item_id": [0]})
    model = RandomRecommender(interactions, n_items=4, seed=7)
    first = model.recommend(1, 2)
    second = model.recommend(1, 2)
    assert first == second
    assert len(first) == 2
    assert 0 not in first


def test_item_knn_recommender_filters_seen():
    interactions = pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1],
            "item_id": [0, 1, 1, 2],
        }
    )
    items = pd.DataFrame({"item_id": [0, 1, 2, 3]})
    model = ItemKNNRecommender.from_interactions(interactions, items, k_neighbors=2)

    recs = model.recommend(0, 2)

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

    recs = model.recommend(0, 2, session_items=[2])

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

    assert model.recommend(0, 1) == [2]
