# Local data

Datasets and generated artifacts live here and are not committed (see `.gitignore`).

```bash
python -m streamlit_recommenders.data.prepare --dataset ml-latest-small   # or ml-32m, goodbooks, ...
python examples/train_baseline_artifacts.py --data data/ml-latest-small   # optional: artifacts/*.npz
```

A prepared folder looks like this; only `items.csv` is required:

```text
data/<dataset-name>/
  items.csv                 item_id, title, image_url, description, ...
  interactions.csv          user_id, item_id, rating, timestamp
  train_interactions.csv    written by train_baseline_artifacts.py
  test_interactions.csv
  artifacts/                itemknn.npz, ease.npz, sequential_cf.npz
  posters/                  TMDB posters (--with-posters)
  dataset.json              completion manifest
```

Options, API keys and Windows notes: [docs/guides/datasets.md](../docs/guides/datasets.md).
Column rules: [docs/concepts/data.md](../docs/concepts/data.md).
