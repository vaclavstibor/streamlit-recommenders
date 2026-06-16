# Showcase — all library features in one demo.
# Run: .venv/bin/streamlit run examples/showcase_demo.py
#      ./scripts/run_demo.sh

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _bootstrap import import_sr

import numpy as np
import pandas as pd

sr = import_sr()

DATA = Path(__file__).parent / "sample_data"
CONFIG = Path(__file__).parent / "showcase_config.yaml"

ITEMS = sr.load_items(str(DATA / "items.csv"))
INTERACTIONS = sr.load_interactions(str(DATA / "interactions.csv"))
USER_EMB = np.load(DATA / "user_emb.npy")
ITEM_EMB = np.load(DATA / "item_emb.npy")
POPULARITY = INTERACTIONS.groupby("item_id").size().reindex(range(len(ITEMS)), fill_value=0).values


def recommend(user_id: int, k: int, alpha: float = 0.5, **params) -> list[int]:
    personal = USER_EMB[user_id] @ ITEM_EMB.T
    scores = alpha * personal + (1 - alpha) * POPULARITY
    seen = set(INTERACTIONS.loc[INTERACTIONS.user_id == user_id, "item_id"])
    ranked = np.argsort(scores)[::-1]
    return [int(i) for i in ranked if i not in seen][:k]


def _user_scores(user_id: int, alpha: float) -> pd.DataFrame:
    personal = USER_EMB[user_id] @ ITEM_EMB.T
    scores = alpha * personal + (1 - alpha) * POPULARITY
    return (
        pd.DataFrame({"item_id": range(len(scores)), "score": scores})
        .merge(ITEMS[["item_id", "title"]], on="item_id")
        .sort_values("score", ascending=False)
    )


def extra_sections() -> None:
    sr.markdown("### Method (inline markdown + LaTeX)")
    sr.markdown(
        r"Blend: $\mathrm{score} = \alpha \cdot u^\top V + (1-\alpha)\cdot\mathrm{pop}$ — "
        r"tweak **α** in the sidebar."
    )

    sr.markdown("### Score plot")
    # alpha read from sidebar happens inside recommend; here we show a static slice for illustration
    sample = _user_scores(0, 0.5).head(12)
    sr.plot(sample, x="title", y="score", title="Top scores (user 0, α=0.5)")

    sr.markdown("### Interactions table")
    sr.table(INTERACTIONS.head(15))

    sr.markdown("### Method")
    sr.markdown_file(Path(__file__).parent / "method.md")


sr.run(
    recommend=recommend,
    items=ITEMS,
    interactions=INTERACTIONS,
    layout="rows",
    config=str(CONFIG),
    title="streamlit_recommenders — Showcase",
    subtitle="Layouts, params, plot, table, markdown, YAML, session clicks",
    params={
        "alpha": sr.slider("alpha", 0.0, 1.0, 0.5),
        "layout": sr.selectbox("Layout", ["rows", "grid", "cards"]),
    },
    body=extra_sections,
)
