# Recommender contract

You own the recommendations; the library owns everything interactive. The entire boundary is one
function.

## The contract

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

| Rule | Detail |
|------|--------|
| Return | Ordered item ids, best first, length ≤ `k` |
| Ids | Must exist in `items`; unknown ids are skipped |
| Params | Sidebar values, minus reserved keys (see [Runtime params](#runtime-params)) |

Optional kwargs are injected **only when your signature accepts them**, so a minimal function stays
minimal:

| Kwarg | Content |
|-------|---------|
| `session_items` | Ordered ids from the session profile (likes/clicks) |
| `selections` | Per-section `{item_id, rank, source}` metadata; swipe dislikes add `sentiment: "dislike"` |
| `hide_seen` | Reserved flag (default `True`); handled for you by `BaseRecommender` |

How the profile becomes a ranking is entirely your model's business — see
**[Feedback & session](feedback.md)**.

## Three ways to plug in

All three return ordered item ids and drop straight into `sr.run(...)`.

### 1. A plain function

Wrap any external scorer, running service, or REST API:

```python
def get_recommendations(user_id, k, session_items=None, **params):
    return my_scorer.rank(user_id, session_items=session_items)[:k]
```

### 2. Subclass `BaseRecommender`

Implement a single `scores()` method; the base class handles seen-filtering and top-`k` ranking:

```python
class MyModel(sr.BaseRecommender):
    def __init__(self, interactions, item_ids):
        self.interactions = interactions
        self.item_ids = item_ids
    def scores(self, user_id, session_items=None, **params):
        return my_score_vector(user_id, session_items)  # one score per item_id
```

### 3. Load exported weights with `ArtifactRecommender`

For a train/export/inspect split, load pure `.npz` arrays — no training code enters the demo:

```python
models = sr.load_artifacts({"EASE": "artifacts/ease.npz"}, interactions=train)
sr.run(get_recommendations=models, items=items, interactions=train)
```

See the **[Training & artifacts guide](../guides/artifacts.md)** for the artifact format.

## The library ships no models

To stay un-opinionated about modeling, the package includes only the `BaseRecommender` contract and
the `ArtifactRecommender` loader. Three reference baselines — `ItemKNNRecommender`,
`EASERecommender`, `SequentialCFRecommender` — live in
[`examples/reference_recommenders.py`](https://github.com/vaclavstibor/streamlit-recommenders/blob/main/examples/reference_recommenders.py)
as copy-and-adapt starting points, not library imports. Heavy training frameworks (RecBole, Cornac,
RecPack, LensKit, Elliot) stay external, reached through this contract rather than added as
dependencies.

## Runtime params

Reserved keys the sidebar consumes instead of forwarding as model params:

| Key | Purpose |
|-----|---------|
| `num_recs` / `k` | Passed as `k` |
| `layout` | `rows` \| `grid` \| `cards` |
| `n_cols` / `n_rows` | Grid dimensions |
| `swipes_per_refresh` | Swipes before the `cards` deck refreshes its queue |
| `hide_seen` | Toggle for already-seen filtering (see [Feedback & session](feedback.md)) |
