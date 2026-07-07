"""Lead demo: compare three trained recommender artifacts in row layout."""

import pandas as pd
import streamlit as st
import streamlit_recommenders as sr

from artifact_recommender import load_artifact_dataset, load_artifact_models

ITEMS, TRAIN, _ = load_artifact_dataset()
models = load_artifact_models(TRAIN)
PARAMS = {
    "num_recs": sr.selectbox("Number of recommendations", [10, 20, 30], index=1),
    "ItemKNN": {
        "history_window": sr.selectbox("History window", ["All", 5, 10, 20], index=0),
    },
}


def intro() -> None:
    sr.markdown(
        "### Interactive Recommender Comparison\n"
        "This page demonstrates the intended use of `streamlit-recommenders` -- "
        "a researcher brings trained models or exported artifacts, while the library "
        "owns the interactive UI, session profile, caching, and side-by-side comparison."
        "Click movies to build a session profile, then use **Get Recommendations** "
        "to refresh all models on the same evidence.\n\n"        
        "#### ItemKNN\n"
        "A classic item-item collaborative filtering baseline. Given a user/session "
        "interaction vector, scores are computed as:"
    )
    st.latex(r"s_u = x_u W_{\mathrm{knn}}")
    sr.markdown(
        "#### EASE\n"
        "A strong shallow linear model for implicit feedback. It uses the same "
        "interface, but the exported weight matrix comes from a regularized closed-form "
        "solution:"
    )
    st.latex(r"s_u = x_u W_{\mathrm{ease}}")
    sr.markdown(
        "#### Sequential CF\n"
        "A lightweight next-item baseline that scores candidates from the last observed "
        "item in the profile:"
    )
    st.latex(r"s_u(i) = P(i \mid i_t)")
    sr.markdown(
        "\n\n"
    )


def appendix() -> None:
    sr.markdown(
        "### Model Artifact Workflow\n"
        "`examples/train_baseline_artifacts.py` prepares a local dataset, trains three "
        "SciPy/NumPy baseline artifacts, and writes `item_ids`, `weights`, and "
        "`popularity` arrays. This demo then loads only those artifacts and exposes "
        "them through `get_recommendations(user_id, k, session_items=...)`."
    )
    summary = pd.DataFrame(
        {
            "model": list(models),
            "artifact": [model.name for model in models.values()],
            "type": [model.model_type for model in models.values()],
            "n_items": [len(model.item_ids) for model in models.values()],
        }
    )
    sr.table(summary)

    current_user = sr.current_user()
    selected = sr.selected_items()
    k = int(sr.param_value("num_recs", 20))
    recs = {
        name: model.get_recommendations(
            current_user,
            k,
            session_items=selected,
            **current_model_params(name),
        )
        for name, model in models.items()
    }
    overlap = sr.recommendation_overlap_matrix(recs)

    sr.markdown(
        "### Recommendation Agreement\n"
        "Jaccard overlap shows whether models converge on similar items "
        "or expose genuinely different recommendation behavior for the current profile. "
        "Darker cells mean stronger agreement between two models."
    )
    sr.plot_overlap_heatmap(overlap, title=f"Top-{k} recommendation overlap")

    sr.markdown(
        "### Top Scores for Current Profile\n"
        "The chart below opens one model's score surface for the active user/session "
        "profile. It is a quick diagnostic for whether a row is driven by specific "
        "evidence or by a popularity-style fallback."
    )
    selected_name, selected_model = next(iter(models.items()))
    scores = selected_model.score_frame(
        current_user,
        session_items=selected,
        items=ITEMS,
        **current_model_params(selected_name),
    ).head(12)
    sr.plot_ranked_items(scores, title=f"{selected_model.name} top scored items")

    sr.dataset_info(ITEMS, TRAIN)


def current_model_params(name: str) -> dict:
    values = {}
    if name == "ItemKNN":
        values["history_window"] = st.session_state.get(
            sr.param_key(f"model.{name}.history_window"),
            "All",
        )
    return values


sr.run(
    get_recommendations=models,
    items=ITEMS,
    interactions=TRAIN,
    params=PARAMS,
    title="3 Models Comparison: Rows Layout",
    subtitle="An interactive movie recommender appendix for comparing trained artifacts through one Streamlit interface.",
    intro=intro,
    body=appendix,
)
