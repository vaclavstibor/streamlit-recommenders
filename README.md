# streamlit_recommenders

Lightweight Streamlit library for **interactive recommender demos**. You load data, bring a trained model or implement `get_recommendations()`, and the library handles UI, cache, session state, baseline comparison, and result inspection.

| [Capabilities](docs/CAPABILITIES.md) | [Contracts](docs/CONTRACTS.md) | [Architecture](docs/ARCHITECTURE.md) |

## Research pitch

Recommender papers and experiment frameworks usually share the same spine: `users`, `items`, `interactions`, a train/test split, a model that returns ordered top-k item ids, baseline comparison, metrics, and enough configuration to reproduce the result. RecBole, Cornac, RecPack, LensKit, and Elliot already cover model training and offline experiments well.

`streamlit-recommenders` focuses on the next step: **standardizing the interactive, user-facing result inspection layer**. A researcher can train a model elsewhere, wrap it as `get_recommendations(user_id, k, session_items=...)`, compare it against strong baselines in stacked rows, click items to simulate a new profile, and add metrics, plots, or markdown appendix notes below the recommendations.

Useful references for the positioning:

- RecBole: unified PyTorch training and evaluation framework ([arXiv 2011.01731](https://arxiv.org/abs/2011.01731), [RecBole 2.0](https://arxiv.org/abs/2206.07351))
- Cornac: multimodal recommender framework for text, images, social data, and interactions ([JMLR 2020](https://www.jmlr.org/papers/volume21/19-805/19-805.pdf))
- LensKit for Python: PyData toolkit for recommender experiments ([ACM CIKM 2020](https://doi.org/10.1145/3340531.3412778))
- RecPack: top-N implicit-feedback experimentation toolkit ([ACM RecSys 2022](https://doi.org/10.1145/3523227.3551472))
- Elliot: rigorous reproducible recommender evaluation framework ([ACM SIGIR 2021](https://doi.org/10.1145/3404835.3463245))
- Baseline rigor: simple tuned baselines often outperform recent neural approaches ([ACM RecSys 2019](https://doi.org/10.1145/3298689.3347058))
- User-centric evaluation: offline metrics should be complemented with interactive/user-facing evidence ([ACM TORS](https://dl.acm.org/doi/10.1145/3800587), [ResQue RecSys 2011](https://doi.org/10.1145/2043932.2043962))

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
.venv/bin/python examples/train_baseline_artifacts.py --data data/ml-32m-filtered
./scripts/run_demo.sh                               # 3 models comparison demo
```

## Minimal demo

```python
import streamlit_recommenders as sr

ITEMS = sr.load_items("data/items.csv")
INTERACTIONS = sr.load_interactions("data/interactions.csv")

def get_recommendations(user_id, k, alpha=0.5, session_items=None, **params):
    ...

sr.run(
    get_recommendations=get_recommendations,
    items=ITEMS,
    interactions=INTERACTIONS,
    layout="rows",
    params={"alpha": sr.slider("alpha", 0.0, 1.0, 0.5)},
)
```

For compare mode, params can be global or scoped to a model label:

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

Missing posters are fine: item cards fall back to a placeholder image. For MovieLens-style local work, keep protected data out of git under `data/ml-32m/`; see `data/README.md`.

## Layouts

All layouts share the same **clickable poster cards** (hover title, description tooltip).

| Layout | Display |
|--------|---------|
| `rows` | One horizontal row with fixed-width cards and side scroll |
| `cards` | Wrapped poster-card gallery for a single recommender |
| `grid` | Catalog-style poster grid with configurable columns |

Compare mode (`get_recommendations={...}`) always uses `rows`.

## Session UX

1. Click item cards to add them to **Selected this session**
2. Click **Get Recommendations** to refresh all compared models with current selections
3. Read state in `body()`: `sr.selected_items()`, `sr.current_user()`, `sr.param_value("alpha")`

## Built-ins

| Area | API |
|------|-----|
| Data | `Dataset`, `ColumnMap`, `load_dataset()`, `validate_dataset()` |
| Recommenders | `ItemKNNRecommender`, `EASERecommender`, `SequentialCFRecommender`, `PopularityRecommender`, `RandomRecommender` |
| Metrics | `evaluate()`, `hit_rate_at_k()`, `recall_at_k()`, `ndcg_at_k()`, `mrr_at_k()`, `coverage()` |
| Viz | `dataset_info()`, `recommendation_overlap_matrix()`, `plot_overlap_heatmap()`, `plot_metric_comparison()`, `plot_ranked_items()`, `plot_score_distribution()` |

## Baseline story

Use three baseline families by default:

- **ItemKNN** for classic item-item collaborative filtering ([Deshpande & Karypis, ACM TOIS 2004](https://doi.org/10.1145/963770.963776)).
- **EASE** for a strong shallow linear implicit-feedback baseline ([Steck, WWW 2019](https://doi.org/10.1145/3308558.3313710)).
- **Sequential CF / SASRec-style** for timestamped next-item behavior ([Kang & McAuley, ICDM 2018](https://arxiv.org/abs/1808.09781)).

The package includes lightweight versions for demos. For full training, use any training code externally and export pure artifacts that match the same `get_recommendations()` contract.

To train the three baseline artifacts and inspect them here:

```bash
pip install -e ".[training]"
.venv/bin/python examples/train_baseline_artifacts.py --data data/<dataset-name>
SR_DATA_DIR=data/<dataset-name> streamlit run examples/3_models_comparison_rows.py
```

The training script reads standard `items.csv`/`interactions.csv`, or raw MovieLens-style `movies.csv`/`ratings.csv`, creates train/test splits if needed, and writes pure `.npz` artifacts for ItemKNN, EASE, and Sequential CF. The Streamlit demo loads only those arrays, not the training code.

## Examples

| File | Pattern |
|------|---------|
| `3_models_comparison_rows.py` | Lead demo: compare ItemKNN, EASE, and Sequential CF artifacts in rows layout |
| `train_baseline_artifacts.py` | Train/export SciPy/NumPy baseline artifacts under `data/<dataset-name>/artifacts` |
| `artifact_recommender.py` | Thin example adapter from exported arrays to `get_recommendations()` |

## Not in scope

Training pipelines, production serving, and heavy model libraries as required dependencies. RecBole, Cornac, RecPack, and LensKit should stay external or optional extras so the core library remains lightweight.

## License

TBD
