# Datasets

Two dataset families are supported out of the box, each one command away. Every prepared folder
lands in the same standard layout (`items.csv`, `interactions.csv`, …) under `data/<dataset-name>/`,
and a `dataset.json` completion manifest prevents accidental re-downloads
(`sr.is_complete(root)`), so preparation is idempotent.

## Keys and paths — `.env` or inline

Keys (`TMDB_API_KEY` / `TMDB_BEARER_TOKEN`, `KAGGLE_USERNAME` / `KAGGLE_KEY`) and the `SR_DATA_DIR`
path can be provided two equivalent ways:

- **`.env` file (recommended).** Copy
  [`.env.example`](https://github.com/vaclavstibor/streamlit-recommenders/blob/main/.env.example) to
  `.env` and fill it in. The preparation CLI loads `.env` automatically:
  ```bash
  python -m streamlit_recommenders.data.prepare --dataset ml-latest-small --with-posters
  ```
- **Inline environment variables.** Prefix the command:
  ```bash
  TMDB_API_KEY=your_key python -m streamlit_recommenders.data.prepare --dataset ml-latest-small --with-posters
  ```

Real environment variables take precedence over `.env`. Only the preparation CLI auto-loads `.env`;
the example apps read `SR_DATA_DIR` from the shell, so export it (`set -a && source .env && set +a`)
or pass it inline (`SR_DATA_DIR=data/ml-latest-small streamlit run ...`).

## Movies — MovieLens

Variants `ml-latest-small`, `ml-latest`, `ml-25m`, `ml-32m`, downloaded directly from GroupLens (no
account needed):

```bash
python -m streamlit_recommenders.data.prepare --dataset ml-latest-small
```

MovieLens ships no artwork. To enrich items with TMDB posters and plot descriptions, get a free API
key at [themoviedb.org](https://www.themoviedb.org/settings/api) and add `--with-posters`:

```bash
TMDB_API_KEY=your_key python -m streamlit_recommenders.data.prepare --dataset ml-latest-small --with-posters
```

`TMDB_BEARER_TOKEN` works as an alternative; `--poster-limit 1000` caps a first run. Enrichment is
robust (retry/backoff, 429 handling) and writes a `metadata_completeness.csv` report of items still
missing a poster or description. Re-running `--with-posters` on a prepared dataset only performs the
enrichment.

## Books — goodbooks-10k

Book covers ship as URLs inside the dataset, so no image enrichment is needed. The download comes
from Kaggle via [`kagglehub`](https://github.com/Kaggle/kagglehub), which ships with the library:

```bash
python -m streamlit_recommenders.data.prepare --dataset goodbooks   # or goodbooks-10k
SR_DATA_DIR=data/goodbooks-10k streamlit run examples/compare_models_rows.py
```

The dataset is public, so `kagglehub` usually needs no credentials. If yours does, authenticate
with `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` / `KAGGLE_KEY`. Alternatively, skip Kaggle by
placing `books.csv` / `ratings.csv` into `data/goodbooks-10k/` yourself.

## From Python

```python
sr.prepare_movielens("ml-32m", with_posters=True)   # TMDB_API_KEY for posters
sr.prepare_goodbooks()
```

## Bring your own domain

Any domain works as long as you produce the standard tables (see the
**[Data contract](../concepts/data.md)**): put `items.csv` and `interactions.csv` in a folder and
point the examples at it with `SR_DATA_DIR=data/your-dataset`. Missing posters fall back to a
placeholder, so identifier-only or sparse-metadata datasets render cleanly too.

Keep protected data out of git under `data/<dataset-name>/`; commit only scripts/docs that describe
how to recreate the local files.
