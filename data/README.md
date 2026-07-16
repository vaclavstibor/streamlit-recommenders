# Local data

This directory is for local datasets and generated artifacts that should not be committed.

Keys and paths can be provided two ways: a `.env` file in the repo root (copy `.env.example`; auto-loaded by the preparation CLI) or inline before the command as shown below. Using `.env` is recommended to keep commands clean. `SR_DATA_DIR` for the example apps must be in the shell environment (export it or pass it inline).

## Movies — MovieLens preprocessing

Use the library CLI to download a MovieLens ZIP, convert it to the local schema, and optionally fetch TMDB metadata/posters:

```bash
python -m streamlit_recommenders.data.prepare --dataset ml-32m
TMDB_API_KEY=... python -m streamlit_recommenders.data.prepare --dataset ml-32m --with-posters
```

Supported datasets are `ml-latest-small`, `ml-latest`, `ml-25m`, and `ml-32m`. The preparation module writes `items.csv`, `interactions.csv`, raw source CSVs, optional poster files, and a completion manifest under `data/<dataset>/`. Poster downloads require `TMDB_API_KEY` or `TMDB_BEARER_TOKEN`; use `--poster-limit 1000` for a smaller first run.

## Books — goodbooks-10k preprocessing

Book covers ship as URLs inside the dataset, so no enrichment step is needed. The download uses Kaggle via `kagglehub` (bundled with the library); alternatively place `books.csv`/`ratings.csv` into `data/goodbooks-10k/` manually:

```bash
python -m streamlit_recommenders.data.prepare --dataset goodbooks
```

The dataset is public, so `kagglehub` usually needs no credentials; if your environment requires them, provide `~/.kaggle/kaggle.json` or set `KAGGLE_USERNAME` and `KAGGLE_KEY`.

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
