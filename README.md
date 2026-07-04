# streamlit_recommenders

Lightweight Streamlit library for **interactive recommender demos**. You load data, implement `recommend()`, the library handles UI, cache, and session state.

| [Capabilities](docs/CAPABILITIES.md) | [Contracts](docs/CONTRACTS.md) | [Architecture](docs/ARCHITECTURE.md) |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
.venv/bin/python examples/generate_sample_data.py   # once
./scripts/run_demo.sh                               # baseline comparison demo
```

## Minimal demo

```python
import streamlit_recommenders as sr

ITEMS = sr.load_items("data/items.csv")
INTERACTIONS = sr.load_interactions("data/interactions.csv")

def recommend(user_id, k, alpha=0.5, session_items=None, **params):
    ...

sr.run(
    recommend=recommend,
    items=ITEMS,
    interactions=INTERACTIONS,
    layout="rows",
    params={"alpha": sr.slider("alpha", 0.0, 1.0, 0.5)},
)
```

## What you write vs. what the library does

| You | Library |
|-----|---------|
| `recommend(user_id, k, **params)` or `RecommenderProtocol` | Adapter, cache, sidebar params |
| `items`, `interactions`, optional `users` / `test` | `Dataset`, validation, user/session UI |
| Optional `body()` callback | Metrics, plots, tables, markdown below recs |
| Dict of models for compare | Stacked rows + per-model **Recommend** |

## Layouts

All layouts share the same **clickable poster cards** (hover title, description tooltip).

| Layout | Display |
|--------|---------|
| `rows` | One horizontal row with fixed-width cards and side scroll |
| `cards` | Same as `rows` |
| `grid` | Multiple rows, 4 fixed-width cards |

Compare mode (`recommend={...}`) always uses `rows`.

## Session UX

1. Click item cards to add them to **Selected this session**
2. Click **Recommend** on a row to refresh that model with current selections
3. Read state in `body()`: `sr.selected_items()`, `sr.current_user()`, `sr.param_value("alpha")`

## Built-ins

| Area | API |
|------|-----|
| Data | `Dataset`, `ColumnMap`, `load_dataset()`, `validate_dataset()` |
| Recommenders | `ItemKNNRecommender`, `EASERecommender`, `SequentialCFRecommender`, `PopularityRecommender`, `RandomRecommender` |
| Metrics | `evaluate()`, `hit_rate_at_k()`, `recall_at_k()`, `ndcg_at_k()`, `mrr_at_k()`, `coverage()` |
| Viz | `plot_metric_comparison()`, `plot_ranked_items()`, `plot_score_distribution()` |

## Examples

| File | Pattern |
|------|---------|
| `baseline_comparison_demo.py` | Custom method vs baselines, metrics, session profile |
| `appendix_demo.py` | Markdown appendix, equations, diagnostics |
| `sequence_cf_demo.py` | Sequence-aware CF baseline and next-item evaluation |

## Not in scope

Training pipelines, production serving, and heavy model libraries as required dependencies. RecPack, Cornac, and LensKit can be wrapped externally or added later as optional extras.

## License

TBD
