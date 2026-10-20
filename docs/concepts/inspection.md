# Inspection views

## What is automatic and what you add

`sr.run` always renders the user picker, profile strips, recommendation sections (one per model),
sidebar controls from `params`, and the seen/fallback labels. Everything else is a helper you call
inside `intro()` (above the app) or `body()` (below it): metric tables, plots, Markdown. The
framework never computes metrics or draws charts on its own. Built-in charts are interactive Plotly
figures; any other Streamlit element works in `body()` too.

## Layouts

| Layout | Shows |
|--------|-------|
| `rows` | one horizontally scrolling row of poster cards per model; used for comparison |
| `grid` | `n_rows` × `n_cols` posters for browsing one model |
| `cards` | one card at a time with Like / Dislike / Skip, refreshed every `swipes_per_refresh` swipes |

Passing several models (`get_recommendations={...}`) always uses `rows`; `cards` is single-model.

## Metrics

Hit rate, recall, NDCG, MRR and catalog coverage over held-out interactions:

```python
recs = {"EASE": {user: ease.get_recommendations(user, 10) for user in test.user_id.unique()}}
metrics = sr.evaluate(recs, test_interactions=test, k=10, all_item_ids=items.item_id)
sr.table(metrics)
sr.plot_metric_comparison(metrics)
```

## Plots

```python
sr.dataset_info(items, train)  # table sizes; genre, activity and rating distributions
sr.plot_overlap_heatmap(sr.recommendation_overlap_matrix(sr.displayed_items()))  # Jaccard overlap of what is on screen
sr.plot_score_distribution(ease.score_frame(sr.current_user()))  # needs a "score" column
```

`sr.displayed_items()` returns exactly the lists on screen, so plots in `body()` match what the
viewer sees.
