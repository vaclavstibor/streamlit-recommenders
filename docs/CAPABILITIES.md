# Capabilities

Practical API overview. Formal contracts: [CONTRACTS.md](CONTRACTS.md).

## Install & run

```bash
pip install -e ".[dev]"
.venv/bin/python examples/train_baseline_artifacts.py --data data/ml-32m-filtered
./scripts/run_demo.sh
```

Requires **Python ≥3.10** and the project venv.

## Public API

| Area | API |
|------|-----|
| App | `sr.run(...)` |
| Data | `load_items`, `load_interactions`, `load_users`, `load_dataset`, `Dataset`, `ColumnMap`, `validate_dataset` |
| Params | `slider`, `selectbox`, `param_value` |
| Layouts | `rows`, `grid`, `cards` |
| Session | `current_user`, `selected_items`, `selections` |
| Content | `plot`, `table`, `markdown`, `markdown_file` |
| Viz helpers | `dataset_info`, `recommendation_overlap_matrix`, `plot_overlap_heatmap`, `plot_metric_comparison`, `plot_ranked_items`, `plot_score_distribution` |
| Metrics | `evaluate`, `hit_rate_at_k`, `recall_at_k`, `ndcg_at_k`, `mrr_at_k`, `coverage` |
| Models | `EmbeddingPopularityRecommender`, `ItemKNNRecommender`, `EASERecommender`, `SequentialCFRecommender`, `PopularityRecommender`, `RandomRecommender` |

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

- All layouts use the same clickable poster cards.
- `rows`: one horizontal row with fixed-width poster cards and side scroll.
- `cards` and `grid`: wrapped poster galleries for browsing/catalog-style display.
- Selected items stay in place as greyed, disabled cards.
- Compare mode always uses `rows`.

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
| `3_models_comparison_rows.py` | Lead demo: compare ItemKNN, EASE, and Sequential CF artifacts in rows layout |
| `train_baseline_artifacts.py` | Train/export SciPy/NumPy baseline artifacts on a local `data/<dataset-name>` folder |
| `artifact_recommender.py` | Thin adapter from exported arrays to `get_recommendations()` |

## Not yet implemented

- Polars backend
- UI file upload for models
- Optional wrappers/examples for RecBole / RecPack / LensKit
