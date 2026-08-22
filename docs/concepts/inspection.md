# Inspection views

Beyond a single ranked list, a demo can show layouts, metrics, and plots. This page draws the line
between what the framework renders automatically and what you add.

## Who provides what

| Layer | Who supplies it | Examples |
|-------|-----------------|----------|
| **Automatic (framework)** | the library | session profile, caching, model adapter, side-by-side comparison, seen-filtering, layouts, sidebar widgets from `params` |
| **Recommender adapter** | you (the model author) | `get_recommendations()` / `scores()` — how a profile becomes a ranking |
| **Optional content** | you (the demo author) | metric tables, score distributions, overlap heatmaps, markdown — via `intro()` / `body()` |

!!! important "Analytical views are helpers you call, not automatic output"
    `sr.evaluate`, `sr.plot_*`, `sr.recommendation_overlap_matrix`, `sr.table`, and `sr.markdown`
    are functions the library **provides**, but they render only where you call them in `body()`.
    The framework does not compute or show them on its own. All built-in charts are interactive
    **Plotly** figures (`st.plotly_chart`), and any Streamlit-compatible chart can be dropped into
    `body()` too.

## Layouts

| Layout | Display |
|--------|---------|
| `rows` | One horizontal row of clickable poster cards with side scroll; drives side-by-side comparison |
| `grid` | Catalog-style clickable poster grid (`n_rows` × `n_cols`) for single-model browsing |
| `cards` | Swipe deck: one card at a time with Like / Dislike / Skip; refreshes after `swipes_per_refresh` swipes |

Compare mode (`get_recommendations={...}`) always uses `rows`; the `cards` deck is single-model.

## Metrics

Ranking metrics over held-out interactions — hit rate, recall, NDCG, MRR, and catalog coverage:

```python
metrics = sr.evaluate({"Ours": recs_by_user}, test_interactions=test, k=10, all_item_ids=items.item_id)
sr.table(metrics)
sr.plot_metric_comparison(metrics)
```

## Plots

```python
sr.dataset_info(items, train)                    # counts, density, distribution tabs
overlap = sr.recommendation_overlap_matrix({"ItemKNN": [1, 2, 3], "EASE": [2, 3, 4]})
sr.plot_overlap_heatmap(overlap)                 # pairwise Jaccard agreement
sr.plot_score_distribution(scores)               # score spread for a model
```

These connect the aggregate numbers reported in a paper to the concrete recommendations a reader
can see. Put them in `body()` so they render beneath the recommendation interface.
