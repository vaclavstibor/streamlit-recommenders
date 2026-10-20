# Recommender contract

The boundary between your model and the library is one function:

```python
def get_recommendations(
    user_id: str | int,
    k: int,
    session_items: list[str | int] | None = None,
    selections: list[dict] | None = None,
    **params,
) -> list[str | int]:
    ...
```

It returns up to `k` item ids, best first. Ids missing from `items` are skipped when rendering.
`params` holds the sidebar values (minus the [reserved keys](#reserved-params)).

Optional arguments are passed only if your signature declares them (or takes `**params`), so a
minimal function stays minimal:

| Argument | Content |
|----------|---------|
| `session_items` | ids liked or clicked in this session, in order |
| `selections` | per-section `{item_id, rank, source}` entries; swipe dislikes carry `sentiment: "dislike"` |
| `hide_seen` | state of the *Hide already-seen items* toggle |

What the model does with them is up to you; [Feedback & session](feedback.md) describes what the
built-in classes do.

## Three ways to plug in

### 1. A plain function

Wrap any scorer, service or API:

```python
def get_recommendations(user_id, k, session_items=None, **params):
    return my_scorer.rank(user_id, session_items=session_items)[:k]
```

### 2. Subclass `BaseRecommender`

Implement `scores()`, returning one score per entry of `self.item_ids`. The base class removes seen
items and takes the top `k`:

```python
class MyModel(sr.BaseRecommender):
    def __init__(self, interactions, item_ids):
        self.interactions = interactions  # used to find seen items
        self.item_ids = item_ids

    def scores(self, user_id, session_items=None, disliked_items=None, **params):
        return my_score_vector(user_id, session_items, disliked_items)
```

`disliked_items` is optional: declare it to receive the session's swipe dislikes.

### 3. Load exported weights with `ArtifactRecommender`

Train elsewhere, export `.npz` arrays, and the demo only loads them:

```python
models = sr.load_artifacts({"EASE": "artifacts/ease.npz"}, interactions=train)
sr.run(get_recommendations=models, items=items, interactions=train)
```

The file format is in [Training & artifacts](../guides/artifacts.md).

## No models in the package

The package contains the contract, `BaseRecommender` and the `ArtifactRecommender` loader, nothing
else. ItemKNN, EASE and Sequential CF are in
[`examples/reference_recommenders.py`](https://github.com/vaclavstibor/streamlit-recommenders/blob/main/examples/reference_recommenders.py)
to copy from. Models trained in RecBole, Cornac, RecPack, LensKit or Elliot connect through the
same function instead of becoming dependencies.

## Reserved params

These `params` keys configure the app and are not forwarded to the model:

| Key | Purpose |
|-----|---------|
| `num_recs` / `k` | number of recommendations, passed as `k` |
| `layout` | `rows`, `grid` or `cards` |
| `n_cols` / `n_rows` | grid size |
| `swipes_per_refresh` | swipes before the `cards` deck refreshes |
