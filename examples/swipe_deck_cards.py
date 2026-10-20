"""Single-model demo: swipe deck (cards layout) with like / dislike / skip feedback.

A Like joins the session profile, a Dislike subtracts the rejected movie's EASE
weight row (so its look-alikes sink), a Skip only marks the card as seen. The
"Dislike weight" slider sets how strongly one dislike counts against one like.
"""

import os

import streamlit_recommenders as sr

DATA_DIR = os.environ.get("SR_DATA_DIR", "data/ml-latest-small")
ITEMS, TRAIN, _ = sr.load_dataset(DATA_DIR)
MODEL = sr.load_artifacts({"EASE": f"{DATA_DIR}/artifacts/ease.npz"}, TRAIN)["EASE"]
PARAMS = {
    "swipes_per_refresh": sr.selectbox("Swipes per refresh", [3, 5, 10], index=1),
    "dislike_weight": sr.slider("Dislike weight", 0.0, 3.0, 1.0, step=0.25),
}


def intro() -> None:
    sr.markdown(
        "### Swipe Deck\n"
        "One card at a time. **Like** adds the movie to your session profile, "
        "**Dislike** pushes down movies similar to it, **Skip** just moves on. "
        "Every few swipes the queue is refreshed against your updated profile.\n\n"
        "Set **Dislike weight** to 0 to compare with a model that only hides disliked "
        "cards. Dislikes alone do not make a profile: a session with dislikes but no "
        "likes gets the popularity fallback, labeled as such."
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
