# Datasets

Every preparation command writes the same layout (`items.csv`, `interactions.csv`, raw files) to
`data/<dataset-name>/` and marks the folder complete in `dataset.json`, so running it again does
nothing unless you pass `--force`.

## MovieLens

`ml-latest-small`, `ml-latest`, `ml-25m` and `ml-32m`, downloaded from GroupLens without an
account:

```bash
python -m streamlit_recommenders.data.prepare --dataset ml-latest-small
```

MovieLens has no images. With a free [TMDB API key](https://www.themoviedb.org/settings/api),
`--with-posters` adds posters and plot descriptions (`--poster-limit 1000` for a shorter first run):

```bash
python -m streamlit_recommenders.data.prepare --dataset ml-latest-small --with-posters
```

Requests are retried on rate limits and network errors, and items still missing a poster or
description are listed in `metadata_completeness.csv`. Adding `--with-posters` to an already
prepared folder rebuilds it with posters, reusing the downloaded files.

## goodbooks-10k

Covers are URLs inside the dataset, so there is no enrichment step. The download goes through
[`kagglehub`](https://github.com/Kaggle/kagglehub), installed with the library:

```bash
python -m streamlit_recommenders.data.prepare --dataset goodbooks
SR_DATA_DIR=data/goodbooks-10k streamlit run examples/compare_models_rows.py
```

The dataset is public and usually needs no Kaggle login; if yours asks for one, use
`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`/`KAGGLE_KEY`. You can also put `books.csv` and
`ratings.csv` into `data/goodbooks-10k/` yourself.

## From Python

```python
sr.prepare_movielens("ml-latest-small", with_posters=True)
sr.prepare_goodbooks()
```

## Keys and environment variables

The preparation CLI reads a `.env` file in the working directory (copy
[`.env.example`](https://github.com/vaclavstibor/streamlit-recommenders/blob/main/.env.example)):
`TMDB_API_KEY` or `TMDB_BEARER_TOKEN`, and optionally `KAGGLE_USERNAME`/`KAGGLE_KEY`. Variables
already set in the shell win over `.env`.

The example apps do not read `.env`; they take `SR_DATA_DIR` from the shell and default to
`data/ml-latest-small`:

```bash
SR_DATA_DIR=data/goodbooks-10k streamlit run examples/compare_models_rows.py
```

!!! note "Windows"
    `VAR=value command` is bash syntax. In PowerShell run `$env:SR_DATA_DIR="data/goodbooks-10k"`
    first, in cmd `set SR_DATA_DIR=data/goodbooks-10k`. Downloads verify HTTPS against the system
    certificate store plus the [certifi](https://pypi.org/project/certifi/) bundle, so the
    `CERTIFICATE_VERIFY_FAILED` error that older releases hit on Windows is fixed without disabling
    verification.

## Your own data

Write `items.csv` and `interactions.csv` following the [Data contract](../concepts/data.md) into
`data/<name>/` and set `SR_DATA_DIR=data/<name>`. Items without images get a placeholder. Keep
datasets and posters out of git; commit only the scripts that recreate them.
