# Getting started

This page takes you from an empty environment to a running, inspectable demo.

## Install

```bash
pip install streamlit-recommenders
```

Requires **Python ≥3.10**. A single install pulls in everything the library and the bundled
examples need (including `scipy` for baseline training and `kagglehub` for the goodbooks
download). Contributors can add the tooling extra:

```bash
pip install "streamlit-recommenders[dev]"   # pytest, build, twine
```

From source:

```bash
git clone https://github.com/vaclavstibor/streamlit-recommenders.git
cd streamlit-recommenders
pip install -e ".[dev]"
```

## Prepare a dataset

Two dataset families work out of the box, each one command away. Every prepared folder ends up in
the same standard layout (`items.csv`, `interactions.csv`, …) under `data/<dataset-name>/`, and a
manifest makes preparation idempotent.

```bash
python -m streamlit_recommenders.data.prepare --dataset ml-latest-small
```

For posters and plot descriptions on MovieLens, add `--with-posters` (needs a free
[TMDB](https://www.themoviedb.org/settings/api) key). Full options — variants, goodbooks, `.env`
keys, bringing your own domain — are in the **[Datasets guide](guides/datasets.md)**.

## Your first demo

Point the library at the prepared folder and pass any callable that returns ranked item ids:

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

Swap the function body for your model — see the **[Recommender contract](concepts/recommender.md)**
for the three ways to plug one in.

## Compare models side by side

Pass a dict of named recommenders; every model receives the same user and session context and
refreshes together. Parameters can be global or scoped to a model label:

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

## Run the bundled examples

The [`examples/`](https://github.com/vaclavstibor/streamlit-recommenders/tree/main/examples)
folder (shipped in the repo, not the wheel) has ready-to-run demos. For a step-by-step, beginner
walkthrough of each one — organized by the three recommender interfaces — see the
**[Examples by interface guide](guides/examples.md)**.

```bash
SR_DATA_DIR=data/ml-latest-small streamlit run examples/reference_recommenders.py
```

| File | Shows |
|------|-------|
| `reference_recommenders.py` | Subclass `BaseRecommender` (ItemKNN / EASE / Sequential CF), fit in memory, compare |
| `compare_models_rows.py` | Compare exported artifacts in the rows layout |
| `swipe_deck_cards.py` | Single-model swipe deck; a feedback-aware model that reads dislikes |
| `compare_models_grid.py` | Grid layout with a sidebar model selector |
| `train_baseline_artifacts.py` | Train/export `.npz` baseline artifacts |
