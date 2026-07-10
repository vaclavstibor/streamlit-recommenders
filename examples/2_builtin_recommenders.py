"""Way 2 demo: fit built-in reference recommenders on interactions (no artifacts).

Each model is fit directly from the interactions table via ``from_interactions``
(the fit step), then compared in stacked rows. This is the path when you want to
train with the library's own recommender definitions instead of loading exported
weights.
"""

import streamlit_recommenders as sr

from artifact_recommender import load_artifact_dataset

ITEMS, TRAIN, _ = load_artifact_dataset()

MODELS = {
    "ItemKNN": sr.ItemKNNRecommender.from_interactions(TRAIN, ITEMS),
    "EASE": sr.EASERecommender.from_interactions(TRAIN, ITEMS),
    "Sequential CF": sr.SequentialCFRecommender.from_interactions(TRAIN, ITEMS),
}


def intro() -> None:
    sr.markdown(
        "### Built-in Recommenders\n"
        "These models are fit in-memory straight from the interactions table with "
        "`sr.<Model>.from_interactions(train, items)` -- no external training or "
        "exported artifacts. Click movies to build a session profile, then use "
        "**Get Recommendations** to refresh all models on the same evidence."
    )


sr.run(
    get_recommendations=MODELS,
    items=ITEMS,
    interactions=TRAIN,
    params={"num_recs": sr.selectbox("Number of recommendations", [10, 20, 30], index=1)},
    title="Built-in Recommenders",
    subtitle="Fit reference baselines on your interactions and compare them in stacked rows.",
    intro=intro,
)
