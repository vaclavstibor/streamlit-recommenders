# Data contract

The data model is pandas-first and intentionally small: it mirrors an offline experiment, so the
tables you already have usually fit as-is.

## Tables

```python
dataset = sr.Dataset(items=items, interactions=train, users=users, test=test)
dataset.validate()
```

| Table | Required | Optional |
|-------|----------|----------|
| `items` | `item_id` | `title`, `image_url`, `description`, domain metadata |
| `users` | `user_id` | segment/profile columns |
| `interactions` / `train` / `test` | `user_id`, `item_id` | `rating`, `timestamp` |

Only `items` is required. `interactions` supplies user history and the user picker; the optional
`train`/`test` splits exist **solely to recompute the offline metrics** shown beside the
recommendations — training never happens inside the demo. `validate_dataset()` checks required
columns and that ids are consistent across tables.

## Item metadata & images

The display layer uses these optional columns when present, and degrades gracefully otherwise:

| Column | Purpose | Fallback |
|--------|---------|----------|
| `title` | Card hover title and profile labels | `item_id` |
| `image_url` | Poster/card image | bundled placeholder |
| `description` | Card tooltip | `title` |

Missing posters are fine — item cards fall back to a bundled placeholder, so a demo renders cleanly
even on identifier-only or sparse-metadata datasets. Extra columns (`genres`, `year`, `tmdb_id`,
`imdb_id`, `poster_path`, …) ride along untouched and are available in your `body()` callback for
plots and diagnostics.

## Column names

Defaults: `user_id`, `item_id`, `rating`, `timestamp`, `title`, `image_url`, `description`. If your
data uses different names, pass a `ColumnMap`. It is honored by validation and item display; the
built-in recommenders and layouts expect the default names, so either rename to the defaults or
pass `item_columns=` to `sr.run` for display.

## Loading a prepared folder

```python
items, train, test = sr.load_dataset("data/ml-latest-small")  # resolves poster paths to local files
```

`load_dataset` expects `items.csv` and either `train_interactions.csv` or `interactions.csv`;
`test_interactions.csv` is optional. Keep local/protected datasets under `data/<dataset-name>/` and
out of git. See the **[Datasets guide](../guides/datasets.md)** to produce these folders.
