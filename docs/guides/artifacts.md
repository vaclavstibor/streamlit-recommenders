# Training & artifacts

The demo does not train models. Train outside, export plain NumPy arrays, and the app only loads
them, so no training code or framework ends up in the demo.

## Export the baselines

```bash
python examples/train_baseline_artifacts.py --data data/ml-latest-small
```

Reads `items.csv`/`interactions.csv` (or raw MovieLens `movies.csv`/`ratings.csv`), writes
`train_interactions.csv`/`test_interactions.csv` if they are missing, and saves
`artifacts/itemknn.npz`, `artifacts/ease.npz` and `artifacts/sequential_cf.npz`.

## Artifact format

An artifact stores item-item weights, not user-item scores or embeddings.

| Key | Shape | Meaning |
|-----|-------|---------|
| `model_type` | `(1,)` string | `"itemknn"`, `"ease"` or `"sequential_cf"`; selects the scoring rule |
| `item_ids` | `(n_items,)` | catalog order of the rows and columns below |
| `weights` | `(n_items, n_items)` float32 | cosine similarities, EASE weights, or row-normalized transitions |
| `popularity` | `(n_items,)` float32 | fallback scores when the profile has no signal |
| `l2` (EASE only) | `(1,)` float32 | regularization, kept for provenance; not read by the loader |

```python
models = sr.load_artifacts({"EASE": "artifacts/ease.npz"}, interactions=train)
```

How `ArtifactRecommender` scores:

- `itemknn`, `ease`: a binary vector over history + session items (optionally limited by a
  `history_window` param) times `weights`.
- `sequential_cf`: the `weights` row of the most recent item.
- Each session dislike subtracts `dislike_weight` × its `weights` row.
- No usable signal: `popularity`, flagged in the UI and by `fallback_reason()`.

Any model that fits these four arrays can be exported the same way and loaded without code changes.

## Bring an externally trained model

Models from RecBole, Cornac, RecPack, LensKit, Elliot or a paper repository need no artifact; wrap
the trained object in the contract:

```python
class TrainedModelAdapter:
    def __init__(self, model):
        self.model = model

    def get_recommendations(self, user_id, k, session_items=None, **params):
        return self.model.rank(user_id, session_items=session_items)[:k]

sr.run(get_recommendations={"Ours": TrainedModelAdapter(model), "EASE": ease}, items=items, interactions=train)
```

## Reference baselines

The same three families as in-app `BaseRecommender` subclasses, in
[`examples/reference_recommenders.py`](https://github.com/vaclavstibor/streamlit-recommenders/blob/main/examples/reference_recommenders.py):

| Family | Class | Reference |
|--------|-------|-----------|
| Item-item CF | `ItemKNNRecommender` | Deshpande & Karypis, ACM TOIS 2004 |
| Linear autoencoder | `EASERecommender` | Steck, WWW 2019 |
| Sequential CF | `SequentialCFRecommender` | first-order item transitions |
