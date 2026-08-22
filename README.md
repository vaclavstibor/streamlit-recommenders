<img src="docs/figures/logo.svg" alt="streamlit-recommenders logo" align="left" width="96" hspace="18"/>

# streamlit-recommenders: Interactive Inspection for Recommender Systems

**Turn a trained recommender into an interactive, inspectable web demo in a few lines of Python.**

[![PyPI](https://img.shields.io/pypi/v/streamlit-recommenders.svg)](https://pypi.org/project/streamlit-recommenders/)
[![Python](https://img.shields.io/pypi/pyversions/streamlit-recommenders.svg)](https://pypi.org/project/streamlit-recommenders/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/vaclavstibor/streamlit-recommenders/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://vaclavstibor.github.io/streamlit-recommenders/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://recommenders-demo.streamlit.app)
[![Video](https://img.shields.io/badge/Video-YouTube-FF0000?logo=youtube&logoColor=white)](https://youtu.be/BvplkLanLwc)

<br clear="left"/>

## Why streamlit-recommenders?

Reproducibility in recommender-systems research usually stops at aggregate offline metrics.
Whether a model actually behaves sensibly — for individual users, across parameter settings,
against baselines — stays invisible unless someone builds a demo, and building demos is
frontend work most researchers would rather skip.

`streamlit-recommenders` removes that barrier. It is a thin presentation layer between your
recommendation model and an interactive [Streamlit](https://streamlit.io) app: you bring a
trained model (or just a function returning ranked item ids), and the library owns the UI,
caching, session state, side-by-side comparison, and light evaluation. It ships **no models** —
what recommends is entirely yours.

- **Inspect per-user behavior** — pick any user, see history and live recommendations, click to simulate session feedback.
- **Probe parameter sensitivity** — expose model parameters as sidebar widgets and watch recommendations react.
- **Compare models side by side** — pass a dict of named models; all receive the same user and session context.
- **Attach light evaluation** — ranking metrics (HR, Recall, NDCG, MRR, coverage), overlap heatmaps, score distributions, markdown/LaTeX below the demo.
- **Inspect honestly** — a seen/unseen toggle, an explicit cold-start seed browser, and a visible popularity-fallback badge mean nothing on screen is silently misattributed to your model.

The goal: make interactive model inspection a standard, low-effort artifact to publish
alongside a paper.

## Install

```bash
pip install streamlit-recommenders
```

Requires Python ≥3.10. See the [Getting started guide](https://vaclavstibor.github.io/streamlit-recommenders/getting-started/)
for source installs and the `[dev]` extra.

## Documentation

Full documentation lives at
[vaclavstibor.github.io/streamlit-recommenders](https://vaclavstibor.github.io/streamlit-recommenders/).
New here? Start with the [Getting started guide](https://vaclavstibor.github.io/streamlit-recommenders/getting-started/).

## Minimal Example

```python
import streamlit_recommenders as sr

ITEMS = sr.load_items("data/items.csv")
INTERACTIONS = sr.load_interactions("data/interactions.csv")
POPULARITY = INTERACTIONS["item_id"].value_counts().index.tolist()

def get_recommendations(user_id, k, session_items=None, **params):
    exclude = set(session_items or [])
    return [item_id for item_id in POPULARITY if item_id not in exclude][:k]

sr.run(
    get_recommendations=get_recommendations,
    items=ITEMS,
    interactions=INTERACTIONS,
    params={"alpha": sr.slider("alpha", 0.0, 1.0, 0.5)},
)
```

```bash
streamlit run app.py
```

Replace the function body with your model and you have an inspectable demo. For **compare
mode**, pass a dict of models (`get_recommendations={"Ours": ours, "EASE": ease}`); params can
be global or scoped to a model label. Your tables follow a small pandas schema — see the
[Data contract](https://vaclavstibor.github.io/streamlit-recommenders/concepts/data/).

## Three Ways to Plug In a Recommender

The library owns the interactive layer; you provide recommendations in whichever form fits:

1. **A plain function** — `def get_recommendations(user_id, k, session_items=None, **params): ...` around any scorer, service, or API.
2. **Subclass `sr.BaseRecommender`** — implement `scores()`; seen-filtering and top-`k` are handled for you.
3. **`sr.ArtifactRecommender`** — load exported `.npz` weights for a train/export/inspect split.

All three return ordered item ids and drop straight into `sr.run(...)`. The package ships no
recommenders of its own; ItemKNN, EASE, and Sequential CF are reference implementations in
[`examples/reference_recommenders.py`](examples/reference_recommenders.py) to copy and adapt.
Heavy training frameworks (RecBole, Cornac, LensKit, …) stay external, reached through the
contract. → [Recommender contract](https://vaclavstibor.github.io/streamlit-recommenders/concepts/recommender/).

## Examples

Examples live in [`examples/`](examples/) (not in the wheel):

| File | Pattern |
|------|---------|
| `reference_recommenders.py` | Subclass `BaseRecommender` (ItemKNN / EASE / Sequential CF), fit in memory, compare |
| `compare_models_rows.py` | Compare exported artifacts in the rows layout |
| `swipe_deck_cards.py` | Single-model swipe deck; a feedback-aware model that reads dislikes |
| `compare_models_grid.py` | Grid layout with a sidebar model selector |
| `train_baseline_artifacts.py` | Train/export `.npz` baseline artifacts |

## Citation

```bibtex
@software{stibor2026streamlitrecommenders,
  author  = {Stibor, V{\'a}clav and Van{\v c}ura, Vojt{\v e}ch and Pe{\v s}ka, Ladislav},
  title   = {StreamlitRecommenders: Towards Recommendation Inspectability as a New Reproducibility Standard},
  year    = {2026},
  url     = {https://github.com/vaclavstibor/streamlit-recommenders},
  version = {0.1.1}
}
```

## License

MIT License — see [LICENSE](LICENSE).
