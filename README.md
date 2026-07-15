# streamlit-recommenders

**Turn a trained recommender into an interactive, inspectable web demo in a few lines of Python.**

[![PyPI](https://img.shields.io/pypi/v/streamlit-recommenders.svg)](https://pypi.org/project/streamlit-recommenders/)
[![Python](https://img.shields.io/pypi/pyversions/streamlit-recommenders.svg)](https://pypi.org/project/streamlit-recommenders/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

## Why streamlit-recommenders?

Reproducibility in recommender systems research usually stops at aggregate offline metrics. Whether a model actually behaves sensibly — for individual users, across parameter settings, against baselines — stays invisible unless someone builds a demo, and building demos is frontend work most researchers don't want to do.

`streamlit-recommenders` is a thin presentation layer between your recommendation model and an interactive [Streamlit](https://streamlit.io) app. You bring a trained model (or just a function returning ranked item ids); the library handles the UI, caching, session state, side-by-side model comparison, and light evaluation:

- **Inspect per-user behavior** — pick any user, see their history and live recommendations, click items to simulate session feedback.
- **Probe parameter sensitivity** — expose model parameters as sidebar widgets and watch recommendations react.
- **Compare models side by side** — pass a dict of named models; all receive the same user and session context.
- **Attach light evaluation** — ranking metrics (HR, Recall, NDCG, MRR, coverage), overlap heatmaps, score distributions, and markdown/LaTeX appendix content below the demo.
- **Stay lightweight** — training frameworks (RecBole, Cornac, LensKit, ...) remain external; the demo loads only your exported artifacts or callable.

The goal: make interactive model inspection a standard, low-effort artifact to publish alongside a paper.

## Installation

```bash
pip install streamlit-recommenders
```

Extras:

```bash
pip install "streamlit-recommenders[training]"   # scipy, for the baseline training example
pip install "streamlit-recommenders[goodbooks]"  # kagglehub, for the goodbooks-10k download
pip install "streamlit-recommenders[dev]"        # pytest, build, twine
```

From source:

```bash
git clone https://github.com/vaclavstibor/streamlit-recommenders.git
cd streamlit-recommenders
pip install -e ".[dev,training]"
```

## Quick start

Create `app.py`:

```python
import streamlit_recommenders as sr

ITEMS = sr.load_items("data/items.csv")
INTERACTIONS = sr.load_interactions("data/interactions.csv")

POPULARITY = INTERACTIONS["item_id"].value_counts().index.tolist()

def get_recommendations(user_id, k, alpha=0.5, session_items=None, **params):
    """Return an ordered list of item ids (here: popularity, minus session items)."""
    exclude = set(session_items or [])
    return [item_id for item_id in POPULARITY if item_id not in exclude][:k]

sr.run(
    get_recommendations=get_recommendations,
    items=ITEMS,
    interactions=INTERACTIONS,
    layout="rows",
    params={"alpha": sr.slider("alpha", 0.0, 1.0, 0.5)},
)
```

Then run:

```bash
streamlit run app.py
```

Replace the function body with your model and you have an inspectable demo. For **compare mode**, pass a dict of models — params can be global or scoped to a model label:

```python
sr.run(
    get_recommendations={"Ours": ours, "EASE": ease},
    items=ITEMS,
    interactions=INTERACTIONS,
    params={
        "num_recs": sr.selectbox("Number of recommendations", [5, 10, 20], index=1),
        "Ours": {"alpha": sr.slider("alpha", 0.0, 1.0, 0.5)},
    },
)
```

## Three ways to plug in a recommender

The library owns the interactive layer; you provide recommendations in whichever of these fits your workflow:

1. **Bring your trained model / weights.** Export your model to plain arrays (or keep the object) and expose `get_recommendations()`. The lead demo uses `sr.ArtifactRecommender`, which loads exported `.npz` weights. Subclass `sr.BaseRecommender` (implement `scores()`) for custom trained models.
2. **Start from a reference baseline.** ItemKNN, EASE, and Sequential CF are defined in `examples/reference_recommenders.py` by subclassing `sr.BaseRecommender` — copy one and fit it in memory with `Model.from_interactions(train, items)`. These are reference implementations to adapt, not library imports.
3. **Thin function adapter.** Skip classes entirely and write a plain `def get_recommendations(user_id, k, session_items=None, **params): ...` around any external scorer or API.

All three return ordered item ids and drop straight into `sr.run(...)`.

## What you write vs. what the library does

| You | Library |
|-----|---------|
| `get_recommendations(user_id, k, **params)` or `RecommenderProtocol` | Adapter, cache, sidebar params, session feedback |
| `items`, `interactions`, optional `users` / `test` | `Dataset`, validation, user/session UI |
| Optional `intro()` / `body()` callbacks | Markdown/math above recs; metrics, plots, tables below recs |
| Dict of models for compare | Stacked rows + shared **Get Recommendations** |

## Data standard

The core data contract is pandas-first and intentionally small:

| Table | Required | Recommended / optional |
|-------|----------|------------------------|
| `items` | `item_id` | `title`, `image_url`, `description`, `genres`, `year`, `tmdb_id`, `imdb_id`, `poster_path` |
| `users` | `user_id` | segment/profile columns |
| `interactions`, `train`, `test` | `user_id`, `item_id` | `rating`, `timestamp` |

Missing posters are fine: item cards fall back to a bundled placeholder image. Keep protected data out of git under `data/<dataset-name>/`; see [data/README.md](https://github.com/vaclavstibor/streamlit-recommenders/blob/main/data/README.md).

## Dataset preparation

Two dataset families are supported out of the box, each one command away. A completion manifest prevents accidental re-downloads, and every prepared folder ends up in the same standard layout (`items.csv`, `interactions.csv`, ...) under `data/<dataset-name>/`.

### Movies — MovieLens

Supported variants: `ml-latest-small`, `ml-latest`, `ml-25m`, `ml-32m` (downloaded directly from GroupLens, no account needed):

```bash
python -m streamlit_recommenders.data.prepare --dataset ml-latest-small
```

MovieLens ships no artwork. To enrich items with TMDB posters and plot descriptions, get a free API key at [themoviedb.org](https://www.themoviedb.org/settings/api) and add `--with-posters`:

```bash
TMDB_API_KEY=your_key python -m streamlit_recommenders.data.prepare --dataset ml-latest-small --with-posters
```

`TMDB_BEARER_TOKEN` is accepted as an alternative; `--poster-limit 1000` caps the download for a quick first run. Re-running with `--with-posters` on an already-prepared dataset only performs the enrichment.

### Books — goodbooks-10k

Book covers ship as URLs inside the dataset itself, so no image enrichment is needed. The
download comes from Kaggle via [`kagglehub`](https://github.com/Kaggle/kagglehub), which is
**not a core dependency** — install it first (otherwise preparation stops with a
`goodbooks-10k not found locally … install kagglehub` message):

```bash
pip install "streamlit-recommenders[goodbooks]"     # installs kagglehub
python -m streamlit_recommenders.data.prepare --dataset goodbooks   # or goodbooks-10k
SR_DATA_DIR=data/goodbooks-10k streamlit run examples/compare_models_rows.py
```

The dataset is public, so `kagglehub` usually needs no credentials. If your environment does
require them (the same way MovieLens posters need `TMDB_API_KEY`), authenticate Kaggle with a
`~/.kaggle/kaggle.json` token or the `KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables.
Alternatively, skip Kaggle entirely by placing `books.csv` / `ratings.csv` into
`data/goodbooks-10k/` yourself.

### Bring your own domain

Any domain works as long as you produce the standard tables (see [Data standard](#data-standard)): put `items.csv` and `interactions.csv` in a folder and point the examples at it with `SR_DATA_DIR=data/your-dataset`.

## Layouts

| Layout | Display |
|--------|---------|
| `rows` | One horizontal row of clickable poster cards with side scroll |
| `grid` | Catalog-style clickable poster grid with configurable rows and columns |
| `cards` | Swipe deck: one card at a time with Like / Dislike / Skip; refreshes after `swipes_per_refresh` swipes |

`rows` and `grid` use clickable poster cards (hover title, description tooltip). Compare mode (`get_recommendations={...}`) always uses `rows`; the `cards` swipe deck is single-model.

## Session UX

1. Click item cards to add them to **Selected this session**; click selected cards again to unselect
2. In the swipe deck, **Like** / **Dislike** / **Skip** each card; dislikes appear in a **Disliked this session** strip
3. Click **Get Recommendations** to refresh all compared models with current selections
4. Read state in `body()`: `sr.selected_items()`, `sr.disliked_items()`, `sr.displayed_items(label)`, `sr.current_user()`, `sr.param_value("alpha")`

## How recommendations are built

The evidence a model scores against is the **user profile** — the union of the selected user's
**history** and their **current-session interactions**:

1. **History.** For a dataset user, their past interactions (`interactions` table) form the
   starting profile. The **Try yourself** session user starts empty, so it is driven purely by
   what you click.
2. **Session interactions.** Clicking a card (or swiping **Like**) adds an item to
   `session_items`; swiping **Dislike** records it with `sentiment: "dislike"`; **Skip** only
   marks the card as seen. These accumulate in the session state as you go.
3. **Refresh.** On **Get Recommendations**, the library passes the merged evidence to your model
   as `session_items` (plus `selections` metadata). It also unions history with session likes to
   avoid re-recommending already-seen items (`effective_seen`). So each refresh folds everything
   you have done so far into the profile.
4. **Scoring & weighting.** How the profile becomes a ranking is your model's business. The
   bundled `ArtifactRecommender` builds a binary profile vector over `history + session_items`
   and multiplies it by the exported weight matrix; the optional **History window** control caps
   how many recent items count, and Sequential CF scores from the last item only.

Feedback is a first-class part of the contract: `session_items` carries likes, `selections`
carries dislike sentiment, and skips are excluded. The reference recommenders only *exclude*
disliked/skipped items, but a custom model can read the dislike sentiment as a negative signal —
see `examples/swipe_deck_cards.py`. Full shapes are in
[docs/CONTRACTS.md](https://github.com/vaclavstibor/streamlit-recommenders/blob/main/docs/CONTRACTS.md).

## Built-ins

| Area | API |
|------|-----|
| Data | `Dataset`, `ColumnMap`, `load_dataset()`, `load_dataset_from_paths()`, `validate_dataset()` |
| Recommenders | `BaseRecommender`, `ArtifactRecommender`, `load_artifacts()` — ItemKNN / EASE / Sequential CF are reference implementations in `examples/reference_recommenders.py`, not library imports |
| Metrics | `evaluate()`, `hit_rate_at_k()`, `recall_at_k()`, `ndcg_at_k()`, `mrr_at_k()`, `coverage()` |
| Viz | `dataset_info()`, `recommendation_overlap_matrix()`, `plot_overlap_heatmap()`, `plot_metric_comparison()`, `plot_ranked_items()`, `plot_score_distribution()` |

## Baseline story

Use three baseline families by default:

- **ItemKNN** for classic item-item collaborative filtering.
- **EASE** for a strong shallow linear implicit-feedback baseline.
- **Sequential CF** for timestamped next-item behavior.

Lightweight reference implementations live in `examples/reference_recommenders.py` (copy-and-adapt, not shipped in the package), and `examples/train_baseline_artifacts.py` exports them as `.npz` artifacts. For full training, use any training code externally and export pure artifacts that match the same `get_recommendations()` contract.

To train the three baseline artifacts and inspect them:

```bash
pip install "streamlit-recommenders[training]"

# Optionally enrich with TMDB posters/descriptions first (see Dataset preparation):
TMDB_API_KEY=your_key python -m streamlit_recommenders.data.prepare --dataset ml-latest-small --with-posters
python examples/train_baseline_artifacts.py --data data/ml-latest-small
SR_DATA_DIR=data/ml-latest-small streamlit run examples/compare_models_rows.py
```

The training script reads standard `items.csv`/`interactions.csv`, or raw MovieLens-style `movies.csv`/`ratings.csv`, creates train/test splits if needed, and writes pure `.npz` artifacts for ItemKNN, EASE, and Sequential CF. The Streamlit demo loads only those arrays, not the training code.

## Examples

Examples live in the [GitHub repository](https://github.com/vaclavstibor/streamlit-recommenders/tree/main/examples) (not in the wheel):

| File | Pattern |
|------|---------|
| `reference_recommenders.py` | Way 2: define models by subclassing `BaseRecommender` (ItemKNN / EASE / Sequential CF), fit in memory, and compare |
| `compare_models_rows.py` | Lead demo (way 1): compare ItemKNN, EASE, and Sequential CF artifacts in rows layout |
| `swipe_deck_cards.py` | Single-model swipe deck (cards layout) with like/dislike/skip; wraps EASE in a feedback-aware model that uses dislikes as a negative signal |
| `compare_models_grid.py` | Grid layout with a sidebar model selector (one artifact at a time) |
| `train_baseline_artifacts.py` | Train/export SciPy/NumPy baseline artifacts under `data/<dataset-name>/artifacts` |

## Citation

If you use `streamlit-recommenders` in your research, please cite:

```bibtex
@software{stibor2026streamlitrecommenders,
  author  = {Stibor, V{\'a}clav and Van{\v c}ura, Vojt{\v e}ch and Pe{\v s}ka, Ladislav},
  title   = {StreamlitRecommenders: Towards Recommendation Inspectability as a New Reproducibility Standard},
  year    = {2026},
  url     = {https://github.com/vaclavstibor/streamlit-recommenders},
  version = {0.1.0}
}
```

## License

Apache License 2.0 — see [LICENSE](https://github.com/vaclavstibor/streamlit-recommenders/blob/main/LICENSE).
