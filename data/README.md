# Local data

*todo: Add preprocessing (including download, posters adding, etc.) one command script, due to rights issues.*

This directory is for local datasets and generated artifacts that should not be committed.

Recommended layout for any local recommender dataset:

```text
data/
  <dataset-name>/
    items.csv
    users.csv
    interactions.csv
    train_interactions.csv
    test_interactions.csv
    artifacts/
      itemknn.npz
      ease.npz
      sequential_cf.npz
    posters/
```

The source can be anywhere on your machine. Copy or transform it into a dataset folder under `data/`, for example `data/ml-32m/`, `data/books-demo/`, or `data/music-demo/`. Do not commit MovieLens files, cached posters, or other protected data.

Minimum schemas:

- `items.csv`: required `item_id`; recommended `title`, `image_url`, `description`; optional movie fields such as `genres`, `year`, `tmdb_id`, `imdb_id`, `poster_path`, `popularity`, `release_date`.
- `users.csv`: required `user_id`; optional profile/segment columns.
- `interactions.csv`, `train_interactions.csv`, `test_interactions.csv`: required `user_id`, `item_id`; optional `rating`, `timestamp`.

Missing `image_url` values are supported. The UI renders a placeholder poster for those items.

`examples/train_baseline_artifacts.py` can create `items.csv`, `interactions.csv`, `train_interactions.csv`, `test_interactions.csv`, and the `artifacts/` folder from either the standard schema or raw MovieLens-style `movies.csv` / `ratings.csv`.
