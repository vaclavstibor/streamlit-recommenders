# Getting started

## Install

```bash
pip install streamlit-recommenders
```

Python 3.10 or newer. The install includes what the examples need (`scipy` for training the
baselines, `kagglehub` for goodbooks). From source, with test tooling:

```bash
git clone https://github.com/vaclavstibor/streamlit-recommenders.git
cd streamlit-recommenders
pip install -e ".[dev]"
```

## Prepare a dataset

```bash
python -m streamlit_recommenders.data.prepare --dataset ml-latest-small
```

This creates `data/ml-latest-small/` with `items.csv` and `interactions.csv`. Posters, other
datasets, API keys and Windows notes are in [Datasets](guides/datasets.md).

## Your first demo

Save as `app.py`:

```python
import streamlit_recommenders as sr

items, train, test = sr.load_dataset("data/ml-latest-small")
POPULARITY = train["item_id"].value_counts().index.tolist()

def get_recommendations(user_id, k, session_items=None, **params):
    exclude = set(session_items or [])
    return [item_id for item_id in POPULARITY if item_id not in exclude][:k]

sr.run(get_recommendations=get_recommendations, items=items, interactions=train)
```

```bash
streamlit run app.py
```

Pick a user in the sidebar (or *Try it yourself* for an empty profile), click posters to add them
to the session, and press **Get Recommendations**. Replace the function body with your model; the
[Recommender contract](concepts/recommender.md) lists the three ways to plug one in.

## Compare models

Pass a dict of named recommenders. All of them get the same user and session and refresh together.
Params can be global or scoped to one model:

```python
sr.run(
    get_recommendations={"Ours": ours, "EASE": ease},
    items=items,
    interactions=train,
    params={
        "num_recs": sr.selectbox("Number of recommendations", [5, 10, 20], index=1),
        "Ours": {"alpha": sr.slider("alpha", 0.0, 1.0, 0.5)},
    },
)
```

`alpha` reaches `ours.get_recommendations(..., alpha=...)` and changes when the slider moves.

Next: the [Examples](guides/examples.md) page runs the bundled demos (in-app baselines, exported
artifacts, the swipe deck).
