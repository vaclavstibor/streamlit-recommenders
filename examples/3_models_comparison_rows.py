"""Lead demo: compare three trained recommender artifacts in row layout."""

import pandas as pd
import streamlit as st
import streamlit_recommenders as sr

from artifact_recommender import load_artifact_dataset, load_artifact_models

ITEMS, TRAIN, TEST = load_artifact_dataset()
models = load_artifact_models(TRAIN)
EVAL_USER_CAP = 500
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


@st.cache_data(show_spinner="Evaluating models on the held-out split...")
def offline_evaluation(k: int) -> pd.DataFrame:
    users = TEST["user_id"].drop_duplicates()
    if len(users) > EVAL_USER_CAP:
        users = users.sample(EVAL_USER_CAP, random_state=0)
    recommendations = {
        name: {user: model.get_recommendations(user, k) for user in users}
        for name, model in models.items()
    }
    held_out = TEST[TEST["user_id"].isin(set(users))]
    return sr.evaluate(recommendations, held_out, k=k, all_item_ids=ITEMS["item_id"].tolist())


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

    with st.expander("Recommendation agreement", expanded=False):
        sr.markdown(
            "Jaccard overlap shows whether models converge on similar items "
            "or expose genuinely different recommendation behavior for the current profile. "
            "Darker cells mean stronger agreement between two models."
        )
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
        sr.plot_overlap_heatmap(overlap, title=f"Top-{k} recommendation overlap")

    with st.expander("Top scores for current profile", expanded=False):
        sr.markdown(
            "Each tab opens one model's score surface for the active user/session "
            "profile. It is a quick diagnostic for whether a row is driven by specific "
            "evidence or by a popularity-style fallback. Raw scores are model-specific "
            "and not comparable across tabs."
        )
        for tab, (name, model) in zip(st.tabs(list(models)), models.items()):
            with tab:
                scores = model.score_frame(
                    current_user,
                    session_items=selected,
                    items=ITEMS,
                    **current_model_params(name),
                ).head(12)
                sr.plot_ranked_items(scores, title=f"{name} top scored items")

    if TEST is not None and len(TEST):
        with st.expander("Offline evaluation on the held-out split", expanded=False):
            sr.markdown(
                "The training script holds out each user's most recent interaction "
                "(leave-last-out). The table below scores all three artifacts against "
                "that split with the library's ranking metrics — the aggregate numbers "
                "that the interactive rows above complement."
            )
            results = offline_evaluation(k)
            pivot = results.pivot(index="metric", columns="model", values="value").round(4)
            sr.table(pivot.reset_index())
            sr.plot_metric_comparison(results, title=f"Ranking metrics @ {k}")
            n_users = min(TEST["user_id"].nunique(), EVAL_USER_CAP)
            st.caption(f"Evaluated on {n_users:,} held-out users.")

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
