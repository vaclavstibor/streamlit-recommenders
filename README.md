<img src="docs/figures/logo.svg" alt="streamlit-recommenders logo" align="left" width="96" hspace="18"/>

# streamlit-recommenders: Interactive Inspection for Recommender Systems

**An interactive, inspectable web demo for your recommender, in a few lines of Python.**

[![PyPI](https://img.shields.io/pypi/v/streamlit-recommenders.svg)](https://pypi.org/project/streamlit-recommenders/)
[![Python](https://img.shields.io/pypi/pyversions/streamlit-recommenders.svg)](https://pypi.org/project/streamlit-recommenders/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/vaclavstibor/streamlit-recommenders/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://vaclavstibor.github.io/streamlit-recommenders/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://recommenders-demo.streamlit.app)
[![Video](https://img.shields.io/badge/Video-YouTube-FF0000?logo=youtube&logoColor=white)](https://youtu.be/BvplkLanLwc)

<br clear="left"/>

Pick a user, see their history and what each model recommends, click or swipe to give feedback,
and watch the recommendations change. `streamlit-recommenders` builds that
[Streamlit](https://streamlit.io) app around your model, so a recommender paper can ship with a
demo instead of only a metrics table.

You supply pandas tables and a function (or object) that returns ranked item ids. The library
handles the UI, session profile, caching, side-by-side comparison, and helpers for metrics and
plots. It contains no models of its own.

- **Per-user behavior:** any user's history next to live recommendations; session clicks, likes and dislikes feed back into the profile.
- **Parameters:** model parameters become sidebar widgets.
- **Comparison:** pass a dict of models; all of them get the same user and session.
- **Evaluation helpers:** hit rate, recall, NDCG, MRR, coverage, overlap heatmaps, score distributions.
- **No hidden substitutes:** cold-start samples and popularity fallbacks are labeled, so nothing on screen is mistaken for the model's output.

## Install

```bash
pip install streamlit-recommenders
```

Python 3.10+. Docs: [vaclavstibor.github.io/streamlit-recommenders](https://vaclavstibor.github.io/streamlit-recommenders/),
starting with [Getting started](https://vaclavstibor.github.io/streamlit-recommenders/getting-started/).

## Minimal example

```python
import streamlit_recommenders as sr

items = sr.load_items("data/items.csv")
interactions = sr.load_interactions("data/interactions.csv")
POPULARITY = interactions["item_id"].value_counts().index.tolist()

def get_recommendations(user_id, k, session_items=None, **params):
    exclude = set(session_items or [])
    return [item_id for item_id in POPULARITY if item_id not in exclude][:k]

sr.run(get_recommendations=get_recommendations, items=items, interactions=interactions)
```

```bash
streamlit run app.py
```

Swap the function body for your model. For comparison, pass `get_recommendations={"Ours": ours, "EASE": ease}`.
Tables follow a small [data contract](https://vaclavstibor.github.io/streamlit-recommenders/concepts/data/).

## Plugging in a model

1. **A plain function** `get_recommendations(user_id, k, session_items=None, **params)` around any scorer, service or API.
2. **A `sr.BaseRecommender` subclass** that implements `scores()`; seen-filtering and top-`k` are done for you.
3. **`sr.ArtifactRecommender`**, which loads exported `.npz` weights, so training stays outside the demo.

ItemKNN, EASE and Sequential CF are in [`examples/reference_recommenders.py`](examples/reference_recommenders.py)
to copy from. See the [recommender contract](https://vaclavstibor.github.io/streamlit-recommenders/concepts/recommender/).

## Examples

In [`examples/`](examples/) (repository only, not in the wheel):

| File | Shows |
|------|-------|
| `reference_recommenders.py` | ItemKNN / EASE / Sequential CF as `BaseRecommender` subclasses, fit in the app |
| `train_baseline_artifacts.py` | training the same baselines and exporting `.npz` artifacts |
| `compare_models_rows.py` | three artifacts side by side |
| `compare_models_grid.py` | one artifact at a time in the grid layout |
| `swipe_deck_cards.py` | EASE as a swipe deck; dislikes push down similar movies |

## Citation

```bibtex
@software{stibor2026streamlitrecommenders,
  author  = {Stibor, V{\'a}clav and Van{\v c}ura, Vojt{\v e}ch and Pe{\v s}ka, Ladislav},
  title   = {StreamlitRecommenders: Towards Recommendation Inspectability as a New Reproducibility Standard},
  year    = {2026},
  url     = {https://github.com/vaclavstibor/streamlit-recommenders},
  version = {0.1.2}
}
```

## License

MIT, see [LICENSE](LICENSE).
