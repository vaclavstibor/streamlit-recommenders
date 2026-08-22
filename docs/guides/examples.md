# Examples by recommender interface

The library gives you **three ways to plug in a recommender** (the
[recommender contract](../concepts/recommender.md)). This page walks each one end-to-end,
beginner-first: which file to open, what to type, and what you'll see. Pick the row that matches
how you already have (or want to build) your model.

| Interface | Best when | Example file | You run |
|-----------|-----------|--------------|---------|
| **1. Plain function** | You already have something that ranks items (a scorer, service, API) | *you write* `app.py` | `streamlit run app.py` |
| **2. Subclass `BaseRecommender`** | You have modeling logic and want seen-filtering + top-`k` handled for you | `examples/reference_recommenders.py` | `streamlit run examples/reference_recommenders.py` |
| **3. `ArtifactRecommender`** | You train elsewhere and want a light, reproducible demo | `examples/train_baseline_artifacts.py` → `examples/compare_models_rows.py` | train once, then `streamlit run …` |

## Prerequisite: prepare the sample dataset (once)

Every example below uses the MovieLens sample. Prepare it a single time:

```bash
python -m streamlit_recommenders.data.prepare --dataset ml-latest-small
```

This creates `data/ml-latest-small/` with `items.csv` and `interactions.csv` in the standard
layout. Full options (posters, goodbooks, your own data) are in the
**[Datasets guide](datasets.md)**.

---

## 1. A plain function — the fastest start

The whole recommender boundary can be a single function that returns ranked item ids. No class, no
files to train — ideal for a first run or for wrapping an existing scorer/service/API.

**Step 1 — create `app.py`** with a tiny popularity recommender:

```python
import streamlit_recommenders as sr

items, train, test = sr.load_dataset("data/ml-latest-small")
POPULARITY = train["item_id"].value_counts().index.tolist()

def get_recommendations(user_id, k, session_items=None, **params):
    exclude = set(session_items or [])
    return [item_id for item_id in POPULARITY if item_id not in exclude][:k]

sr.run(get_recommendations=get_recommendations, items=items, interactions=train)
```

**Step 2 — run it:**

```bash
streamlit run app.py
```

**What you'll see:** a user picker (including the empty **Try yourself** user), that user's history,
and a row of recommendations. Click a poster to add it to the session profile and watch the list
react.

**Next:** swap the function body for your own model — anything callable that returns ordered item
ids drops straight in. → [Recommender contract §1](../concepts/recommender.md#1-a-plain-function).

---

## 2. Subclass `BaseRecommender` — bring a model, skip the plumbing

Implement a single `scores()` method (one score per item); the base class handles seen-filtering
and top-`k` ranking for you. `examples/reference_recommenders.py` does exactly this for three
classic baselines — **ItemKNN**, **EASE**, and **Sequential CF** — and compares them.

**Step 1 — run it** (data prepared above; `SR_DATA_DIR` defaults to `data/ml-latest-small`):

```bash
streamlit run examples/reference_recommenders.py
```

**What happens:** the three baselines are fit **in memory** at startup, then shown side by side on
the same user and session context.

**Where to look in the file:** each class implements only `scores(user_id, session_items, **params)`
— that is the entire model. Everything interactive (profile, caching, comparison, layouts) is the
library's job. Copy one class as the starting point for your own model.

!!! warning "Low on memory?"
    Fitting EASE / ItemKNN in memory builds a dense *item × item* matrix (~0.4 GB each for the
    MovieLens sample). On a machine with under ~2 GB free this can be OOM-killed. If that happens,
    use the artifact path in **[section 3](#3-artifactrecommender-train-once-inspect-anywhere)** —
    it trains once outside the app and the demo only loads small arrays.

**Next:** → [Recommender contract §2](../concepts/recommender.md#2-subclass-baserecommender).

---

## 3. `ArtifactRecommender` — train once, inspect anywhere

Keep training **out of the demo**: train externally, export plain `.npz` arrays, and let the app
load only those. This is the lightest, most reproducible way to publish a demo.

**Step 1 — train and export the baseline artifacts** (writes `itemknn.npz`, `ease.npz`,
`sequential_cf.npz` under `data/ml-latest-small/artifacts/`):

```bash
python examples/train_baseline_artifacts.py --data data/ml-latest-small
```

**Step 2 — inspect them.** Rows layout:

```bash
streamlit run examples/compare_models_rows.py
```

…or the same three models in a grid with a sidebar selector:

```bash
streamlit run examples/compare_models_grid.py
```

**What you'll see:** the three models loaded straight from disk — **no training code runs at demo
time** — compared on shared user/session context.

**Where to look in the file:** a single call does the loading —
`sr.load_artifacts({"ItemKNN": ".../itemknn.npz", …}, TRAIN)`. The `.npz` format is documented in
the **[Training & artifacts guide](artifacts.md)**.

### Going further: a feedback-aware custom model

`examples/swipe_deck_cards.py` shows the flexible edge of the same contract: a small
`FeedbackAwareEASE` class wraps an EASE artifact and reads swipe **dislikes** (`selections` with
`sentiment: "dislike"`) as a *negative* signal, demoting look-alike items — in the `cards` (swipe)
layout.

```bash
streamlit run examples/swipe_deck_cards.py
```

It needs `ease.npz` from Step 1. This is a template for consuming negative feedback in your own
model. → [Feedback & session](../concepts/feedback.md).

---

## Already have a trained model elsewhere?

You do not need artifacts at all. Train in RecBole, Cornac, LensKit, or a paper repo, then expose
the object through the same contract — see
**[Bring an externally trained model](artifacts.md#bring-an-externally-trained-model)**.
