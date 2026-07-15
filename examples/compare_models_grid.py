"""Compare trained artifacts one at a time in the catalog-style grid layout.

Compare mode (a dict of models) always renders stacked rows, so this demo
uses a sidebar model selector instead: pick one artifact and inspect its
recommendations as a grid. The selected model name is passed through
``params`` so switching models refreshes the displayed recommendations.
"""

import os

import streamlit as st
import streamlit_recommenders as sr

DATA_DIR = os.environ.get("SR_DATA_DIR", "data/ml-latest-small")
ITEMS, TRAIN, _ = sr.load_local_dataset(DATA_DIR)
models = sr.load_artifacts(
    {
        "ItemKNN": f"{DATA_DIR}/artifacts/itemknn.npz",
        "EASE": f"{DATA_DIR}/artifacts/ease.npz",
        "Sequential CF": f"{DATA_DIR}/artifacts/sequential_cf.npz",
    },
    TRAIN,
)

st.sidebar.subheader("Model")
model_name = st.sidebar.selectbox("Model", list(models), label_visibility="collapsed")
model = models[model_name]


def intro() -> None:
    sr.markdown(
        "### Grid Inspection, One Model at a Time\n"
        "The `grid` layout shows a catalog-style view of a single model's "
        "recommendations. Switch the model in the sidebar to compare how "
        "ItemKNN, EASE, and Sequential CF fill the same grid for the same "
        "user and session profile. Click items to build session evidence, "
        "then refresh."
    )


def appendix() -> None:
    current_user = sr.current_user()
    selected = sr.selected_items()
    # A single model renders under one section; read the ids it actually shows
    # so the score chart matches the grid (including the cold-start sample).
    displayed = set(next(iter(sr.displayed_items().values()), []))
    scores = model.score_frame(
        current_user,
        session_items=selected,
        items=ITEMS,
    )
    scores = scores[scores["item_id"].isin(displayed)].head(12)
    sr.plot_ranked_items(scores, title=f"{model_name} displayed items by score")
    sr.dataset_info(ITEMS, TRAIN)


sr.run(
    get_recommendations=model,
    items=ITEMS,
    interactions=TRAIN,
    layout="grid",
    params={
        "n_rows": sr.selectbox("Rows", [1, 2, 3, 4, 5], index=2),
        "n_cols": sr.selectbox("Columns", [5, 8, 10, 12, 15], index=2),
        # Plain value, not a widget: it feeds the run-context hash so changing
        # the sidebar model invalidates previously displayed recommendations.
        "model": model_name,
    },
    title=f"Models Comparison: Grid Layout ({model_name})",
    subtitle="Inspect one artifact at a time in a catalog-style grid; switch models in the sidebar.",
    intro=intro,
    body=appendix,
)
