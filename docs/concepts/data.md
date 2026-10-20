# Data contract

Plain pandas tables, shaped like an offline experiment, so existing data usually fits as is.

## Tables

| Table | Required columns | Optional columns |
|-------|------------------|------------------|
| `items` | `item_id` | `title`, `image_url`, `description`, any metadata |
| `interactions` / `train` / `test` | `user_id`, `item_id` | `rating`, `timestamp` |
| `users` | `user_id` | any profile columns |

Only `items` is required. `interactions` provides user history and the user picker; without it the
sidebar picks an item id as context instead. `test` is for your own evaluation with
[`sr.evaluate`](inspection.md#metrics). No training happens inside the demo.

```python
dataset = sr.Dataset(items=items, interactions=train, users=users, test=test)
dataset.validate()  # required columns + every referenced id exists
```

## Display columns

| Column | Used for | If missing |
|--------|----------|------------|
| `title` | card hover text, profile strips | the `item_id` |
| `image_url` | poster | bundled placeholder image |
| `description` | card hover text | the title |

Other columns (`genres`, `year`, `tmdb_id`, …) are ignored by the UI and remain available to your
`body()` code.

## Other column names

The defaults are `user_id`, `item_id`, `rating`, `timestamp`, `title`, `image_url`, `description`.
`ColumnMap` maps other names for validation (`sr.Dataset`, `sr.load_dataset_from_paths`), and
`sr.run(item_columns={"id": ..., "title": ..., "image": ..., "description": ...})` does the same for
the cards. The recommender classes and the session logic read `user_id`/`item_id` directly, so
renaming the DataFrame columns is usually simpler.

## Prepared folders

```python
items, interactions, test = sr.load_dataset("data/ml-latest-small")
```

Reads `items.csv`, then `train_interactions.csv` if it exists, otherwise `interactions.csv`, plus
an optional `test_interactions.csv`. Relative poster paths are resolved against the folder. The
[Datasets guide](../guides/datasets.md) shows how to create such folders.
