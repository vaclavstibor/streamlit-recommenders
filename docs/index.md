# streamlit-recommenders

**Turn a trained recommender into an interactive, inspectable web demo in a few lines of Python.**

`streamlit-recommenders` is a thin presentation layer between your recommendation model and an
interactive [Streamlit](https://streamlit.io) app. You bring a function (or object) that returns
ranked item ids; the library owns the UI, caching, session profile, side-by-side comparison, and
light evaluation. It ships **no models** — what recommends is entirely yours.

![System overview](figures/architecture.svg)

## Why inspect, not just measure?

Reproducibility in recommender-systems research usually stops at aggregate offline metrics.
Whether a model actually behaves sensibly — for individual users, across parameter settings,
against baselines — stays invisible unless someone builds a demo, and building demos is frontend
work most researchers would rather skip. This library removes that barrier so an inspectable demo
can ship alongside a paper.

## 60-second quickstart

```python
import streamlit_recommenders as sr

ITEMS = sr.load_items("data/items.csv")
INTERACTIONS = sr.load_interactions("data/interactions.csv")
POPULARITY = INTERACTIONS["item_id"].value_counts().index.tolist()

def get_recommendations(user_id, k, session_items=None, **params):
    exclude = set(session_items or [])
    return [item_id for item_id in POPULARITY if item_id not in exclude][:k]

sr.run(get_recommendations=get_recommendations, items=ITEMS, interactions=INTERACTIONS)
```

```bash
streamlit run app.py
```

Replace the function body with your model and you have an inspectable demo.

## Where to next

- **[Getting started](getting-started.md)** — install, prepare a dataset, and run your first demo.
- **Concepts** — the ideas the library is built on:
    - **[Recommender contract](concepts/recommender.md)** — the one function to implement, three ways to plug in.
    - **[Data contract](concepts/data.md)** — the small pandas schema your tables follow.
    - **[Feedback & session](concepts/feedback.md)** — how the profile, likes/dislikes, seen-filtering, and fallbacks work.
    - **[Inspection views](concepts/inspection.md)** — layouts, metrics, and plots, and who provides what.
    - **[Architecture](concepts/architecture.md)** — how a demo flows through the library.
- **Guides** — task-focused walk-throughs:
    - **[Datasets](guides/datasets.md)** — MovieLens & goodbooks preparation, TMDB posters, bring your own.
    - **[Training & artifacts](guides/artifacts.md)** — export `.npz` baselines and the artifact format.
- **[API reference](api.md)** — generated from the source docstrings.
