# Researcher: load data, embeddings, and custom scoring.
# Library: UI, cache, layout, sliders — no streamlit import.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _bootstrap import import_sr

import numpy as np
import pandas as pd

sr = import_sr()

DATA = Path(__file__).parent / "sample_data"

ITEMS = pd.read_csv(DATA / "items.csv")
INTERACTIONS = pd.read_csv(DATA / "interactions.csv")
USER_EMB = np.load(DATA / "user_emb.npy")
ITEM_EMB = np.load(DATA / "item_emb.npy")

POPULARITY = INTERACTIONS.groupby("item_id").size().reindex(range(len(ITEMS)), fill_value=0).values


def recommend(user_id: int, k: int, alpha: float = 0.5) -> list[int]:
    personal = USER_EMB[user_id] @ ITEM_EMB.T
    scores = alpha * personal + (1 - alpha) * POPULARITY
    seen = set(INTERACTIONS.loc[INTERACTIONS.user_id == user_id, "item_id"])
    ranked = np.argsort(scores)[::-1]
    return [int(i) for i in ranked if i not in seen][:k]


def show_analysis() -> None:
    sr.markdown("### Method")
    sr.markdown(
        r"Scores blend personal embedding dot-product with popularity: "
        r"$\alpha \cdot u^\top V + (1-\alpha) \cdot \mathrm{pop}$."
    )
    chart = pd.DataFrame({"item_id": range(len(POPULARITY)), "popularity": POPULARITY})
    sr.plot(chart, x="item_id", y="popularity", title="Item popularity")


sr.run(
    recommend=recommend,
    items=ITEMS,
    interactions=INTERACTIONS,
    layout="rows",
    title="Embedding Demo",
    subtitle="Blend personal embeddings with popularity — tune α live.",
    params={
        "alpha": sr.slider("alpha", 0.0, 1.0, 0.5),
        "num_recs": sr.selectbox("num_recs", [5, 10, 15]),
    },
    body=show_analysis,
)
