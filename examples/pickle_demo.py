# Researcher: joblib.load + thin recommend() wrapper.
# Library: sidebar, layout, session state.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _bootstrap import import_sr

import joblib
import pandas as pd

sr = import_sr()

DATA = Path(__file__).parent / "sample_data"

ITEMS = pd.read_csv(DATA / "items.csv")
INTERACTIONS = pd.read_csv(DATA / "interactions.csv")
MODEL = joblib.load(DATA / "model.pkl")


def recommend(user_id: int, k: int, **params) -> list[int]:
    return MODEL.recommend(user_id, k, **params)


sr.run(
    recommend=recommend,
    items=ITEMS,
    interactions=INTERACTIONS,
    layout="grid",
    title="Pickle Model Demo",
    subtitle="Popularity-based recommender loaded from joblib.",
    params={"num_recs": sr.selectbox("num_recs", [5, 10, 20])},
)
