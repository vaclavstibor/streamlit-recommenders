"""Interactive recommendations with a paper-style markdown appendix."""

from pathlib import Path

import numpy as np
import streamlit_recommenders as sr
from streamlit_recommenders.recommenders import EmbeddingPopularityRecommender

DATA = Path(__file__).parent / "sample_data"
CONFIG = Path(__file__).parent / "appendix_config.yaml"

ITEMS = sr.load_items(str(DATA / "items.csv"))
TRAIN = sr.load_interactions(str(DATA / "train_interactions.csv"))
USER_EMB = np.load(DATA / "user_emb.npy")
ITEM_EMB = np.load(DATA / "item_emb.npy")

model = EmbeddingPopularityRecommender.from_interactions(USER_EMB, ITEM_EMB, ITEMS, TRAIN)


def appendix() -> None:
    alpha = float(sr.param_value("alpha", 0.5))
    user_id = sr.current_user()
    selected = sr.selected_items()
    scores = model.score_frame(user_id, alpha, session_items=selected)

    sr.markdown_file(Path(__file__).parent / "appendix.md")
    sr.markdown("### Diagnostics")
    sr.plot_ranked_items(scores.head(12), title="Top ranked items")
    sr.plot_score_distribution(scores, title="All candidate scores")


sr.run(
    recommend=model,
    items=ITEMS,
    interactions=TRAIN,
    layout="cards",
    config=str(CONFIG),
    title="Markdown Appendix Demo",
    subtitle="Keep paper details, equations, and diagnostics next to the live recommender.",
    params={"alpha": sr.slider("alpha", 0.0, 1.0, 0.5)},
    body=appendix,
)
