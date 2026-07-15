"""Single-model demo: swipe deck (cards layout) with like / dislike / skip feedback.

The reference recommenders only *exclude* disliked items from the candidate pool.
``FeedbackAwareEASE`` below shows the other half of the contract: a custom model
can read the ``selections`` feedback the library already passes it and treat
dislikes as a negative signal, pushing down items similar to what was rejected --
without changing any reference recommender.
"""

import os

import numpy as np
import streamlit_recommenders as sr


class FeedbackAwareEASE:
    """Wrap an EASE artifact and use swipe dislikes as a negative signal."""

    def __init__(self, base, penalty: float = 1.0) -> None:
        self.base = base
        self.penalty = penalty

    def get_recommendations(
        self,
        user_id,
        k,
        session_items=None,
        selections=None,
        **params,
    ):
        disliked = [
            entry["item_id"]
            for entry in (selections or [])
            if entry.get("sentiment") == "dislike"
        ]
        scores = self.base.scores(user_id, session_items=session_items, **params).astype(float).copy()
        for item_id in disliked:
            index = self.base.item_index.get(item_id)
            if index is not None:
                # An EASE weight row is the similarity of the disliked item to
                # every item, so subtracting it demotes look-alikes.
                scores -= self.penalty * self.base.weights[index]
        seen = set(session_items or []) | set(disliked)
        ranked = (self.base.item_ids[i] for i in np.argsort(scores)[::-1])
        return [item_id for item_id in ranked if item_id not in seen][:k]


DATA_DIR = os.environ.get("SR_DATA_DIR", "data/ml-latest-small")
ITEMS, TRAIN, _ = sr.load_local_dataset(DATA_DIR)
MODEL = FeedbackAwareEASE(
    sr.load_artifacts({"EASE": f"{DATA_DIR}/artifacts/ease.npz"}, TRAIN)["EASE"]
)
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
        "This demo wraps EASE in a small `FeedbackAwareEASE` model that reads the "
        "`selections` feedback (including `sentiment: \"dislike\"`) the library passes it "
        "and pushes down items similar to the ones you rejected -- a template for using "
        "negative feedback in your own model. Dislikes also show up in the "
        "**Disliked this session** strip above.\n\n"
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
