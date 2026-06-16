"""Generate synthetic sample data for examples."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent / "sample_data"
ROOT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
n_users, n_items, dim = 8, 24, 16

items = pd.DataFrame(
    {
        "item_id": range(n_items),
        "title": [f"Item {i}" for i in range(n_items)],
        "image_url": [
            f"https://picsum.photos/seed/item{i}/200/300" for i in range(n_items)
        ],
        "description": [f"Description for item {i}" for i in range(n_items)],
        "category": rng.choice(["A", "B", "C"], size=n_items),
    }
)
items.to_csv(ROOT / "items.csv", index=False)

interactions = []
for user_id in range(n_users):
    seen = rng.choice(n_items, size=6, replace=False)
    for item_id in seen:
        interactions.append({"user_id": user_id, "item_id": int(item_id), "rating": rng.integers(1, 6)})
pd.DataFrame(interactions).to_csv(ROOT / "interactions.csv", index=False)

user_emb = rng.normal(size=(n_users, dim))
item_emb = rng.normal(size=(n_items, dim))
np.save(ROOT / "user_emb.npy", user_emb)
np.save(ROOT / "item_emb.npy", item_emb)

scores_rows = []
for user_id in range(n_users):
    raw = user_emb[user_id] @ item_emb.T
    for item_id, score in enumerate(raw):
        scores_rows.append({"user_id": user_id, "item_id": item_id, "score": float(score)})
pd.DataFrame(scores_rows).to_csv(ROOT / "scores.csv", index=False)

try:
    import joblib

    class SimpleRecommender:
        def __init__(self, interactions_df: pd.DataFrame):
            self.interactions_df = interactions_df
            self.popularity = interactions_df.groupby("item_id").size()

        def recommend(self, user_id, k, **params):
            seen = set(
                self.interactions_df.loc[
                    self.interactions_df.user_id == user_id, "item_id"
                ]
            )
            ranked = self.popularity.sort_values(ascending=False)
            return [int(i) for i in ranked.index if i not in seen][:k]

    interactions_df = pd.read_csv(ROOT / "interactions.csv")
    joblib.dump(SimpleRecommender(interactions_df), ROOT / "model.pkl")
except ImportError:
    pass

print(f"Sample data written to {ROOT}")
