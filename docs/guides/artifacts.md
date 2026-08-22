# Training & artifacts

The runtime does not train heavy models. The recommended pattern is **train externally → export
pure arrays → inspect** — the demo loads only the arrays, never the training code. This keeps the
boundary explicit and the runtime lightweight.

## Export baseline artifacts

One optional script trains three baseline families and writes NumPy artifacts:

```bash
python examples/train_baseline_artifacts.py --data data/ml-latest-small
SR_DATA_DIR=data/ml-latest-small streamlit run examples/compare_models_rows.py
```

It reads standard `items.csv`/`interactions.csv` (or raw MovieLens `movies.csv`/`ratings.csv`),
creates train/test splits if needed, and writes `artifacts/itemknn.npz`, `artifacts/ease.npz`, and
`artifacts/sequential_cf.npz`.

## Artifact format

Each `.npz` is a small set of arrays. **It stores item–item weights, not user–item scores and not
embeddings** — scoring happens at runtime as `profile_vector @ weights`, with a popularity fallback
when the profile has no signal.

| Key | Shape / dtype | Meaning |
|-----|---------------|---------|
| `model_type` | string tag | `"itemknn"` / `"ease"` / `"sequential_cf"` — selects the scoring branch |
| `item_ids` | `(n_items,)` | Catalog order; indexes the rows/cols of `weights` and `popularity` |
| `weights` | `(n_items, n_items)` float32 | Item–item matrix: kNN cosine similarity, EASE closed-form weights, or row-normalized transitions |
| `popularity` | `(n_items,)` float32 | Global popularity, used as the fallback vector |
| `l2` *(EASE only)* | `(1,)` float32 | Regularization strength; recorded for provenance, not read by the loader |

Load exported artifacts with `sr.load_artifacts({label: path}, interactions=train)` (see the
**[Recommender contract](../concepts/recommender.md)**). At inference `ArtifactRecommender` builds a
binary profile vector over `history + session_items` (capped by the optional **History window**
control) and multiplies by `weights`; sequential models score from the last item's weight row.
`fallback_reason()` reports when a call returns popularity instead of the model's own output — the
UI surfaces this as a badge (see **[Feedback & session](../concepts/feedback.md)**).

## Bring an externally trained model

You do not need artifacts at all. Train in RecBole, Cornac, RecPack, LensKit, Elliot, or a paper
repo, then expose the trained object through the contract:

```python
class TrainedModelAdapter:
    def get_recommendations(self, user_id, k, session_items=None, **params):
        return self.model.rank(user_id, session_items=session_items)[:k]

sr.run(get_recommendations={"Ours": TrainedModelAdapter(), "EASE": ease}, items=items, interactions=train)
```

## Reference baselines

Defined in
[`examples/reference_recommenders.py`](https://github.com/vaclavstibor/streamlit-recommenders/blob/main/examples/reference_recommenders.py)
(subclassing `BaseRecommender`), not shipped by the library — copy one as a starting point:

| Family | Reference class | Canonical citation |
|--------|-----------------|--------------------|
| Item-item CF | `ItemKNNRecommender` | Deshpande & Karypis, ACM TOIS 2004 |
| Shallow linear CF | `EASERecommender` | Steck, WWW 2019 |
| Sequential CF | `SequentialCFRecommender` | first-order item transitions |
