# Local data

This directory is for local datasets and generated artifacts that should not be committed.

## MovieLens preprocessing

Use the preprocessing script to download a MovieLens ZIP, convert it to the local schema, and optionally fetch TMDB metadata/posters:

```bash
python scripts/preprocess_movielens.py --dataset ml-32m
TMDB_API_KEY=... python scripts/preprocess_movielens.py --dataset ml-32m --with-tmdb --download-posters --force
```

Supported datasets are `ml-latest-small`, `ml-latest`, `ml-25m`, and `ml-32m`. The script writes `items.csv`, `interactions.csv`, raw source CSVs, and optional poster files under `data/<dataset>/`. Poster downloads require `TMDB_API_KEY` or `TMDB_BEARER_TOKEN`; use `--poster-limit 1000` for a smaller first run.

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

`examples/train_baseline_artifacts.py` can create `train_interactions.csv`, `test_interactions.csv`, and the `artifacts/` folder from the generated standard schema.
