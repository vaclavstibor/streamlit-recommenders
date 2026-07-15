"""Compare trained artifacts one at a time in the catalog-style grid layout.

Compare mode (a dict of models) always renders stacked rows, so this demo
uses a sidebar model selector instead: pick one artifact and inspect its
recommendations as a grid. The selected model name is passed through
``params`` so switching models refreshes the displayed recommendations.
"""

import streamlit as st
import streamlit_recommenders as sr

from artifact_recommender import load_artifact_dataset, load_artifact_models

ITEMS, TRAIN, _ = load_artifact_dataset()
models = load_artifact_models(TRAIN)

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
    scores = model.score_frame(
        current_user,
        session_items=selected,
        items=ITEMS,
    ).head(12)
    sr.plot_ranked_items(scores, title=f"{model_name} top scored items")
    sr.dataset_info(ITEMS, TRAIN)


sr.run(
    get_recommendations=model,
    items=ITEMS,
    interactions=TRAIN,
    layout="grid",
    params={
        "num_recs": sr.selectbox("Number of recommendations", [10, 20, 30], index=1),
        # Plain value, not a widget: it feeds the run-context hash so changing
        # the sidebar model invalidates previously displayed recommendations.
        "model": model_name,
    },
    title=f"Models Comparison: Grid Layout ({model_name})",
    subtitle="Inspect one artifact at a time in a catalog-style grid; switch models in the sidebar.",
    intro=intro,
    body=appendix,
)
