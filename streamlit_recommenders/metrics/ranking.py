from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import pandas as pd


def hit_rate_at_k(recs: Sequence, ground_truth: Sequence, k: int) -> float:
    truth = set(ground_truth)
    if not truth:
        return 0.0
    return float(bool(set(_top_k(recs, k)) & truth))


def recall_at_k(recs: Sequence, ground_truth: Sequence, k: int) -> float:
    truth = set(ground_truth)
    if not truth:
        return 0.0
    return len(set(_top_k(recs, k)) & truth) / len(truth)


def mrr_at_k(recs: Sequence, ground_truth: Sequence, k: int) -> float:
    truth = set(ground_truth)
    if not truth:
        return 0.0
    for rank, item_id in enumerate(_top_k(recs, k), start=1):
        if item_id in truth:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(recs: Sequence, ground_truth: Sequence, k: int) -> float:
    truth = set(ground_truth)
    if not truth:
        return 0.0
    dcg = 0.0
    for rank, item_id in enumerate(_top_k(recs, k), start=1):
        if item_id in truth:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(truth), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def coverage(recommendations: Mapping | Sequence[Sequence], all_item_ids: Sequence) -> float:
    catalog = set(all_item_ids)
    if not catalog:
        return 0.0
    rec_lists = recommendations.values() if isinstance(recommendations, Mapping) else recommendations
    recommended = {item_id for recs in rec_lists for item_id in recs}
    return len(recommended & catalog) / len(catalog)


def evaluate(
    recommendations: Mapping,
    test_interactions: pd.DataFrame,
    *,
    k: int,
    item_col: str = "item_id",
    user_col: str = "user_id",
    all_item_ids: Sequence | None = None,
) -> pd.DataFrame:
    """Evaluate one or more recommenders against held-out interactions."""
    truth = _truth_by_user(test_interactions, user_col, item_col)
    model_recs = _normalize_recommendations(recommendations)
    rows = []

    for model_name, recs_by_user in model_recs.items():
        per_user = {
            "hit_rate": [],
            "recall": [],
            "ndcg": [],
            "mrr": [],
        }
        for user_id, relevant in truth.items():
            recs = recs_by_user.get(user_id, [])
            per_user["hit_rate"].append(hit_rate_at_k(recs, relevant, k))
            per_user["recall"].append(recall_at_k(recs, relevant, k))
            per_user["ndcg"].append(ndcg_at_k(recs, relevant, k))
            per_user["mrr"].append(mrr_at_k(recs, relevant, k))

        for metric_name, values in per_user.items():
            rows.append(
                {
                    "model": model_name,
                    "metric": metric_name,
                    "k": k,
                    "value": sum(values) / len(values) if values else 0.0,
                }
            )
        if all_item_ids is not None:
            rows.append(
                {
                    "model": model_name,
                    "metric": "coverage",
                    "k": k,
                    "value": coverage(recs_by_user, all_item_ids),
                }
            )

    return pd.DataFrame(rows)


def _top_k(recs: Sequence, k: int) -> list:
    result = []
    seen = set()
    for item_id in recs:
        if item_id in seen:
            continue
        seen.add(item_id)
        result.append(item_id)
        if len(result) >= k:
            break
    return result


def _truth_by_user(df: pd.DataFrame, user_col: str, item_col: str) -> dict:
    if df.empty:
        return {}
    missing = [col for col in [user_col, item_col] if col not in df.columns]
    if missing:
        raise ValueError(f"test_interactions is missing columns: {missing}")
    return df.groupby(user_col)[item_col].apply(list).to_dict()


def _normalize_recommendations(recommendations: Mapping) -> dict[str, dict]:
    if not recommendations:
        return {}
    first_value = next(iter(recommendations.values()))
    if isinstance(first_value, Mapping):
        return {str(model): dict(recs) for model, recs in recommendations.items()}
    return {"recommendations": dict(recommendations)}
