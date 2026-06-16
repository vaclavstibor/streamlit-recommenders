# Researcher: precomputed scores + lookup in recommend().
# Library: YAML config for params, top-scores table.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _bootstrap import import_sr

import pandas as pd

sr = import_sr()

DATA = Path(__file__).parent / "sample_data"

ITEMS = pd.read_csv(DATA / "items.csv")
INTERACTIONS = pd.read_csv(DATA / "interactions.csv")
SCORES = pd.read_csv(DATA / "scores.csv")


def recommend(user_id: int, k: int, min_score: float = 0.0, **params) -> list[int]:
    seen = set(INTERACTIONS.loc[INTERACTIONS.user_id == user_id, "item_id"])
    user_scores = SCORES.loc[SCORES.user_id == user_id].copy()
    user_scores = user_scores[~user_scores.item_id.isin(seen)]
    user_scores = user_scores[user_scores.score >= min_score]
    return user_scores.sort_values("score", ascending=False).item_id.head(k).tolist()


def show_scores() -> None:
    sr.markdown("### Top scores (sample)")
    sr.table(SCORES.head(20))


sr.run(
    recommend=recommend,
    items=ITEMS,
    interactions=INTERACTIONS,
    layout="cards",
    title="Precomputed Matrix Demo",
    subtitle="Lookup precomputed user–item scores and filter by threshold.",
    config=str(Path(__file__).parent / "demo_config.yaml"),
    params={"min_score": sr.slider("min_score", -5.0, 5.0, 0.0)},
    body=show_scores,
)
