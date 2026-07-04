# Capabilities

Practical API overview. Formal contracts: [CONTRACTS.md](CONTRACTS.md).

## Install & run

```bash
pip install -e ".[dev]"
.venv/bin/python examples/generate_sample_data.py
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
| Viz helpers | `plot_metric_comparison`, `plot_ranked_items`, `plot_score_distribution` |
| Metrics | `evaluate`, `hit_rate_at_k`, `recall_at_k`, `ndcg_at_k`, `mrr_at_k`, `coverage` |
| Models | `EmbeddingPopularityRecommender`, `ItemKNNRecommender`, `EASERecommender`, `SequentialCFRecommender`, `PopularityRecommender`, `RandomRecommender` |

## `sr.run()`

```python
sr.run(
    recommend=fn_or_model_or_dict,
    items=items,
    interactions=train,
    layout="rows",
    params={"alpha": sr.slider("alpha", 0.0, 1.0, 0.5)},
    config="baseline_config.yaml",
    body=lambda: sr.table(metrics),
)
```

Pass a dict for compare mode:

```python
sr.run(recommend={"Ours": ours, "EASE": ease}, items=items, interactions=train)
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

## Item cards and layouts

- All layouts use the same clickable poster cards.
- `rows` and `cards`: one horizontal row with fixed-width poster cards and side scroll.
- `grid`: wrapped rows of 4 fixed-width poster cards.
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

## Examples

| File | Shows |
|------|-------|
| `baseline_comparison_demo.py` | Custom method vs ItemKNN/EASE/popularity/random + metrics |
| `appendix_demo.py` | Markdown appendix, equations, score diagnostics |
| `sequence_cf_demo.py` | Timestamped interactions + sequence baseline |

## Not yet implemented

- Polars backend
- UI file upload for models
- Optional wrappers for RecPack / Cornac / LensKit
