"""Single-model demo: swipe deck (cards layout) with like/dislike feedback."""

import streamlit_recommenders as sr

from artifact_recommender import load_artifact_dataset, load_artifact_models

ITEMS, TRAIN, _ = load_artifact_dataset()
MODEL = load_artifact_models(TRAIN)["EASE"]
PARAMS = {
    "swipes_per_refresh": sr.selectbox("Swipes per refresh", [3, 5, 10], index=1),
}


def intro() -> None:
    sr.markdown(
        "### Swipe Deck\n"
        "One card at a time. **Like** adds the movie to your session profile, "
        "**Dislike** records negative feedback and drops the card, **Skip** just moves on. "
        "After the configured number of swipes the queue is refreshed against your updated "
        "profile, so recommendations adapt as you go.\n\n"
        "Use the sidebar to tune swipes per refresh."
    )


sr.run(
    get_recommendations=MODEL,
    items=ITEMS,
    interactions=TRAIN,
    params=PARAMS,
    layout="cards",
    title="Swipe Deck: Cards Layout",
    subtitle="Like / dislike movies one card at a time and watch recommendations adapt.",
    intro=intro,
)
