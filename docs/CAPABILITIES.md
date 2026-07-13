# Capabilities

Practical API overview. Formal contracts: [CONTRACTS.md](CONTRACTS.md).

## Install & run

```bash
pip install streamlit-recommenders          # or, from a clone: pip install -e ".[dev,training]"
python -m streamlit_recommenders.data.prepare --dataset ml-latest-small # or ml-latest, ml-25m, ml-32m; add --with-posters (needs TMDB_API_KEY)
python examples/train_baseline_artifacts.py --data data/ml-latest-small # Additional training script for baseline models and exporting artifacts
SR_DATA_DIR=data/ml-latest-small streamlit run examples/3_models_comparison_rows.py # 3 models comparison (in rows layout) library demonstration
```

Requires **Python ≥3.10**.

## Public API

| Area | API |
|------|-----|
| App | `sr.run(...)` |
| Data | `load_items`, `load_interactions`, `load_users`, `load_dataset`, `load_local_dataset`, `resolve_image_urls`, `Dataset`, `ColumnMap`, `validate_dataset` |
| Data prep | `prepare_movielens`, `prepare_goodbooks`, `is_complete` |
| Params | `slider`, `selectbox`, `param_value` |
| Layouts | `rows`, `grid`, `cards` |
| Session | `current_user`, `selected_items`, `selections` |
| Content | `plot`, `table`, `markdown`, `markdown_file` |
| Viz helpers | `dataset_info`, `recommendation_overlap_matrix`, `plot_overlap_heatmap`, `plot_metric_comparison`, `plot_ranked_items`, `plot_score_distribution` |
| Metrics | `evaluate`, `hit_rate_at_k`, `recall_at_k`, `ndcg_at_k`, `mrr_at_k`, `coverage` |
| Models | `ItemKNNRecommender`, `EASERecommender`, `SequentialCFRecommender`, `ArtifactRecommender`, `BaseRecommender` |

## `sr.run()`

```python
sr.run(
    get_recommendations=fn_or_model_or_dict,
    items=items,
    interactions=train,
    layout="rows",
    params={"alpha": sr.slider("alpha", 0.0, 1.0, 0.5)},
    intro=lambda: sr.markdown("### Model contract"),
    body=lambda: sr.table(metrics),
)
```

Pass a dict for compare mode:

```python
sr.run(get_recommendations={"Ours": ours, "EASE": ease}, items=items, interactions=train)
```

Compare mode shows one shared **Get Recommendations** button so all rows refresh
against the same session profile.

Model-specific controls are keyed by recommender label:

```python
sr.run(
    get_recommendations={"Ours": ours, "EASE": ease},
    items=items,
    interactions=train,
    params={"Ours": {"alpha": sr.slider("alpha", 0.0, 1.0, 0.5)}},
)
```

## Data

```python
dataset = sr.load_dataset(
    items="items.csv",
    interactions="train_interactions.csv",
    users="users.csv",
    test="test_interactions.csv",
)
```

Default logical columns: `user_id`, `item_id`, `rating`, `timestamp`, `title`, `image_url`, `description`.

Only `item_id` is required for `items`. Movie demos should prefer `title`, `image_url`, and `description`; missing images render with a placeholder. Local protected datasets should live under `data/` and stay out of git.

Load a prepared folder (resolves poster paths to local files) with:

```python
items, train, test = sr.load_local_dataset("data/ml-latest-small")
```

## Data preparation

Prepare datasets from inside the library (no standalone scripts required):

```python
sr.prepare_movielens("ml-32m")                      # items.csv + interactions.csv
sr.prepare_movielens("ml-32m", with_posters=True)   # + TMDB descriptions/posters (TMDB_API_KEY)
sr.prepare_goodbooks()                              # books carry cover image URLs (kagglehub optional)
```

Or via CLI:

```bash
python -m streamlit_recommenders.data.prepare --dataset ml-32m --with-posters
```

Each prepared folder gets a `dataset.json` manifest; `sr.is_complete(root)` guards against
re-running collection, so preparation is idempotent and a finished dataset is not re-fetched
before a demo run. TMDB enrichment is robust (retry/backoff, 429 handling) and writes a
`metadata_completeness.csv` report listing items still missing a poster or description.

## External model training

The core package does not train heavy models. Train with any external code, export pure artifacts, then pass a function/object that returns ordered item ids:

```python
def get_recommendations(user_id, k, session_items=None, **params):
    return trained_model.rank(user_id, session_items=session_items)[:k]

sr.run(
    get_recommendations={"Ours": get_recommendations, "EASE": ease_baseline},
    items=items,
    interactions=train,
)
```

One optional script trains the three default baseline families and exports NumPy artifacts:

```bash
pip install -e ".[training]"
.venv/bin/python examples/train_baseline_artifacts.py --data data/<dataset-name>
SR_DATA_DIR=data/<dataset-name> streamlit run examples/3_models_comparison_rows.py
```

The script writes `artifacts/itemknn.npz`, `artifacts/ease.npz`, and `artifacts/sequential_cf.npz`. Each artifact contains `item_ids`, `weights`, and `popularity`; the demo loads only these arrays and never imports the training code.

Recommended baseline story:

| Family | Built-in class | Use |
|--------|----------------|-----|
| ItemKNN | `ItemKNNRecommender` | classic item-item CF baseline |
| EASE | `EASERecommender` | strong shallow linear implicit-feedback baseline |
| Sequential CF | `SequentialCFRecommender` | timestamped next-item baseline; compare to SASRec-style models |

## Item cards and layouts

- `rows`: one horizontal row with fixed-width clickable poster cards and side scroll.
- `grid`: wrapped clickable poster gallery for catalog-style browsing.
- `cards`: swipe deck. One card at a time with Like / Dislike / Skip buttons.
- Clickable layouts (`rows`, `grid`): selected items stay in place as greyed cards; click again to unselect.
- Compare mode always uses `rows`.

### Swipe deck (`cards`)

Like adds the item to the session profile (same signal as a card click); Dislike records
negative feedback and removes the card; Skip just advances. After `swipes_per_refresh`
like/dislike swipes the queue is refreshed against the updated profile:

```python
sr.run(get_recommendations=model, items=items, interactions=train, layout="cards", swipes_per_refresh=5)
```

`swipes_per_refresh` defaults to 5 and can also be overridden via a reserved sidebar param.
Swipe is single-model; compare mode (dict input) stays on `rows`.

## Metrics

```python
metrics = sr.evaluate(
    {"Ours": {0: [1, 2, 3]}},
    test_interactions=test,
    k=10,
    all_item_ids=items.item_id,
)
sr.table(metrics)
sr.plot_metric_comparison(metrics)
```

## Dataset Inspection

```python
sr.dataset_info(items, train)
```

The reusable dataset block summarizes item/interactions tables and exposes simple distribution plots for interactions per user, genres, and categorical item metadata.

## Recommendation Agreement

```python
overlap = sr.recommendation_overlap_matrix({"ItemKNN": [1, 2, 3], "EASE": [2, 3, 4]})
sr.plot_overlap_heatmap(overlap, title="Top-k recommendation overlap")
```

The overlap helper computes pairwise Jaccard agreement between model outputs and renders it as a compact heatmap for compare-mode demos.

## Examples

| File | Shows |
|------|-------|
| `2_builtin_recommenders.py` | Way 2: fit built-in baselines on interactions (no artifacts) and compare |
| `3_models_comparison_rows.py` | Lead demo (way 1): compare ItemKNN, EASE, and Sequential CF artifacts in rows layout |
| `4_swipe_deck_cards.py` | Single-model swipe deck (cards layout) with like/dislike/skip and auto-refresh |
| `train_baseline_artifacts.py` | Train/export SciPy/NumPy baseline artifacts on a local `data/<dataset-name>` folder |
| `artifact_recommender.py` | Example glue: `load_artifact_models` + re-export of `sr.ArtifactRecommender` (I/O lives in `sr.data`) |

## Not yet implemented

- Polars backend
- UI file upload for models
- Optional wrappers/examples for RecBole / RecPack / LensKit
