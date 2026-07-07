import pandas as pd

from streamlit_recommenders.metrics import (
    coverage,
    evaluate,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from streamlit_recommenders.viz.plot import recommendation_overlap_matrix


def test_ranking_metrics_on_small_example():
    recs = [3, 2, 1]
    truth = [1, 4]

    assert hit_rate_at_k(recs, truth, 2) == 0.0
    assert hit_rate_at_k(recs, truth, 3) == 1.0
    assert recall_at_k(recs, truth, 3) == 0.5
    assert mrr_at_k(recs, truth, 3) == 1 / 3
    assert round(ndcg_at_k(recs, truth, 3), 4) == 0.3066


def test_coverage():
    assert coverage({1: [1, 2], 2: [2, 3]}, [1, 2, 3, 4]) == 0.75


def test_evaluate_multiple_models():
    test = pd.DataFrame({"user_id": [1, 1, 2], "item_id": [3, 4, 5]})
    recs = {
        "A": {1: [3, 2], 2: [1, 5]},
        "B": {1: [1, 2], 2: [5, 6]},
    }

    result = evaluate(recs, test, k=2, all_item_ids=[1, 2, 3, 4, 5, 6])

    assert set(result["model"]) == {"A", "B"}
    assert {"hit_rate", "recall", "ndcg", "mrr", "coverage"} <= set(result["metric"])


def test_recommendation_overlap_matrix():
    overlap = recommendation_overlap_matrix(
        {
            "A": [1, 2, 3],
            "B": [2, 3, 4],
            "C": [5],
        }
    )

    assert overlap.loc["A", "A"] == 1.0
    assert overlap.loc["A", "B"] == 0.5
    assert overlap.loc["A", "C"] == 0.0
