"""Next-item style demo using sequence-aware collaborative filtering."""

from pathlib import Path

import streamlit_recommenders as sr
from streamlit_recommenders.recommenders import PopularityRecommender, SequentialCFRecommender

DATA = Path(__file__).parent / "sample_data"
CONFIG = Path(__file__).parent / "sequence_config.yaml"

ITEMS = sr.load_items(str(DATA / "items.csv"))
TRAIN = sr.load_interactions(str(DATA / "train_interactions.csv"))
TEST = sr.load_interactions(str(DATA / "test_interactions.csv"))

sequence = SequentialCFRecommender.from_interactions(TRAIN, ITEMS)
popularity = PopularityRecommender.from_interactions(TRAIN)
models = {
    "Sequence CF": sequence,
    "Popularity": popularity,
}


def diagnostics() -> None:
    k = int(sr.param_value("num_recs", 20))
    users = sorted(TEST["user_id"].unique().tolist())
    recs = {
        name: {user_id: model.recommend(user_id, k) for user_id in users}
        for name, model in models.items()
    }
    metrics = sr.evaluate(recs, TEST, k=k, all_item_ids=ITEMS["item_id"].tolist())

    sr.markdown("### Next-item evaluation")
    sr.plot_metric_comparison(metrics, title=f"Sequence metrics at k={k}")
    sr.table(metrics)

    sr.markdown("### Recent training events")
    sr.table(TRAIN.sort_values(["user_id", "timestamp"]).tail(20))


sr.run(
    recommend=models,
    items=ITEMS,
    interactions=TRAIN,
    layout="rows",
    config=str(CONFIG),
    title="Sequence CF Demo",
    subtitle="A lightweight transition baseline for sequence-like interaction logs.",
    body=diagnostics,
)
