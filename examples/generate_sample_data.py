"""Generate synthetic movie-like sample data for examples."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent / "sample_data"
ROOT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
n_users, n_items, dim = 10, 30, 16
genres = ["Action", "Drama", "Sci-Fi", "Comedy", "Documentary"]

items = pd.DataFrame(
    {
        "item_id": range(n_items),
        "title": [f"Movie {i:02d}" for i in range(n_items)],
        "image_url": [
            f"https://picsum.photos/seed/item{i}/200/300" for i in range(n_items)
        ],
        "description": [
            f"A {genres[i % len(genres)].lower()} recommendation candidate used in the demo catalog."
            for i in range(n_items)
        ],
        "genre": [genres[i % len(genres)] for i in range(n_items)],
    }
)
items.to_csv(ROOT / "items.csv", index=False)

users = pd.DataFrame(
    {
        "user_id": range(n_users),
        "label": [f"Research user {i}" for i in range(n_users)],
    }
)
users.to_csv(ROOT / "users.csv", index=False)

interactions = []
for user_id in range(n_users):
    seen = rng.choice(n_items, size=8, replace=False)
    for step, item_id in enumerate(seen):
        interactions.append(
            {
                "user_id": user_id,
                "item_id": int(item_id),
                "rating": int(rng.integers(1, 6)),
                "timestamp": user_id * 100 + step,
            }
        )
interactions_df = pd.DataFrame(interactions)
train = interactions_df.groupby("user_id").head(6).reset_index(drop=True)
test = interactions_df.groupby("user_id").tail(2).reset_index(drop=True)
train.to_csv(ROOT / "interactions.csv", index=False)
train.to_csv(ROOT / "train_interactions.csv", index=False)
test.to_csv(ROOT / "test_interactions.csv", index=False)

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

print(f"Sample data written to {ROOT}")
