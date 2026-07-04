"""Compare a custom method against common baselines."""

from pathlib import Path

import numpy as np
import streamlit_recommenders as sr
from streamlit_recommenders.recommenders import (
    EASERecommender,
    EmbeddingPopularityRecommender,
    ItemKNNRecommender,
    PopularityRecommender,
    RandomRecommender,
)

DATA = Path(__file__).parent / "sample_data"
CONFIG = Path(__file__).parent / "baseline_config.yaml"

ITEMS = sr.load_items(str(DATA / "items.csv"))
TRAIN = sr.load_interactions(str(DATA / "train_interactions.csv"))
TEST = sr.load_interactions(str(DATA / "test_interactions.csv"))
USER_EMB = np.load(DATA / "user_emb.npy")
ITEM_EMB = np.load(DATA / "item_emb.npy")

ours = EmbeddingPopularityRecommender.from_interactions(USER_EMB, ITEM_EMB, ITEMS, TRAIN)
models = {
    "Our method": ours,
    "ItemKNN": ItemKNNRecommender.from_interactions(TRAIN, ITEMS, k_neighbors=8),
    "EASE": EASERecommender.from_interactions(TRAIN, ITEMS, l2=100.0),
    "Popularity": PopularityRecommender.from_interactions(TRAIN),
    "Random": RandomRecommender(TRAIN, n_items=len(ITEMS), seed=42),
}


def appendix() -> None:
    alpha = float(sr.param_value("alpha", 0.5))
    k = int(sr.param_value("num_recs", 20))
    users = sorted(TEST["user_id"].unique().tolist())
    recs = {
        name: {user_id: model.recommend(user_id, k, alpha=alpha) for user_id in users}
        for name, model in models.items()
    }
    metrics = sr.evaluate(recs, TEST, k=k, all_item_ids=ITEMS["item_id"].tolist())

    sr.markdown("### Offline baseline check")
    sr.table(metrics)
    sr.plot_metric_comparison(metrics, title=f"Baseline metrics at k={k}")

    sr.markdown("### Current score explanation")
    user_id = sr.current_user()
    selected = sr.selected_items()
    scores = ours.score_frame(user_id, alpha, session_items=selected)
    sr.plot_ranked_items(scores.head(12), title=f"Our method top scores (alpha={alpha})")


sr.run(
    recommend=models,
    items=ITEMS,
    interactions=TRAIN,
    layout="rows",
    config=str(CONFIG),
    title="Baseline Comparison Demo",
    subtitle="Compare a custom recommender against ItemKNN, EASE, popularity, and random baselines.",
    params={"alpha": sr.slider("alpha", 0.0, 1.0, 0.5)},
    body=appendix,
)
