# Interfaces & contracts

Formal data shapes and signatures. Overview: [CAPABILITIES.md](CAPABILITIES.md).

## Data

`Dataset` is a small pandas bundle:

```python
sr.Dataset(
    items=items,
    interactions=train,
    users=users,
    test=test,
)
```

| Table | Required columns | Optional columns |
|-------|------------------|------------------|
| `items` | `item_id` | `title`, `image_url`, `description`, domain metadata |
| `users` | `user_id` | segment/profile columns |
| `interactions` / `train` / `test` | `user_id`, `item_id` | `rating`, `timestamp` |

Use `ColumnMap` when your data uses different names. `validate_dataset()` checks required columns and id consistency. Note: custom `ColumnMap`s are honored by validation and item display, but not automatically threaded into the built-in recommenders and layouts — pass `item_columns=` to `sr.run` for display names, or rename columns to the defaults for the built-ins.

### Item metadata

Only `item_id` is required. The display layer uses the following optional columns when present:

| Column | Purpose | Fallback |
|--------|---------|----------|
| `title` | Card hover title and profile labels | `item_id` |
| `image_url` | Poster/card image | built-in placeholder |
| `description` | Card tooltip | `title` |

Movie-like datasets can carry extra metadata without changing the core contract, for example `genres`, `year`, `tmdb_id`, `imdb_id`, `poster_path`, `popularity`, or `release_date`. Keep these as ordinary DataFrame columns and use them in `body()` for plots, diagnostics, or markdown appendix sections.

For local MovieLens-derived work, use a gitignored path such as:

```text
data/
  <dataset-name>/
    items.csv
    users.csv
    interactions.csv
    train_interactions.csv
    test_interactions.csv
    posters/
```

Do not commit MovieLens data or cached posters unless their license explicitly allows it. Commit only scripts/docs that describe how to recreate the local files.

### Preparation

`sr.prepare_movielens(dataset, root=None, *, with_posters=False, tmdb_api_key=None, poster_limit=0, force=False)` and `sr.prepare_goodbooks(root=None, *, force=False)` write `items.csv`/`interactions.csv` plus a `dataset.json` manifest. `sr.is_complete(root)` reports whether a folder is fully prepared; preparation is a no-op on a complete folder unless `force=True` (or posters are newly requested). `sr.load_dataset(root)` returns `(items, train_or_interactions, test)` with poster paths resolved to local files (missing ones fall back to the placeholder).

## Recommender

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

Accepted shapes: plain function, callable, object with `.get_recommendations()`, or `{label: model}` for compare mode.

| Rule | Detail |
|------|--------|
| Return | Ordered item ids, best first, length ≤ `k` |
| Ids | Must exist in `items`; unknown ids are skipped |
| Params | Sidebar values except reserved keys |

Optional injected kwargs when accepted by signature:

| Kwarg | Content |
|-------|---------|
| `session_items` | Ordered ids from the session profile (likes/clicks) |
| `selections` | Per-section `{item_id, rank, source}` metadata; swipe dislikes add `sentiment: "dislike"` |

### Feedback signals

The three swipe/click actions map onto the contract as follows, so a custom model can use as
much of the feedback as it wants:

| Action | Carried as | Reference models | A custom model can |
|--------|-----------|-----------------|--------------------|
| Like / click | `session_items` (+ a `selections` entry) | Added to the positive profile vector | Same |
| Dislike | `selections` entry with `sentiment: "dislike"` | Excluded from the candidate pool only | Read the sentiment and treat it as a negative signal |
| Skip | Excluded ids (not in `selections`) | Excluded so the card does not reappear | — |

The reference recommenders only *exclude* disliked/skipped items; they do not use dislikes as a
negative training signal. Because `selections` is passed through untouched, a custom model that
overrides `get_recommendations(..., selections=...)` can read the dislike sentiment and act on
it — see `examples/swipe_deck_cards.py::FeedbackAwareEASE` for a worked example. `sr.disliked_items()`
exposes the current session's disliked ids to `body()`, and the runner renders them in a
"Disliked this session" strip.

Use `effective_seen(interactions, user_id, session_items)` for filtering. Artifact paths, model checkpoints, external API handles, or framework-specific objects are construction details of the adapter, not part of the core recommender protocol.

### External training frameworks

Heavy training frameworks are intentionally outside the core dependency set. Train in RecBole, Cornac, RecPack, LensKit, Elliot, or a paper repository, then expose the trained artifact through this contract:

```python
class TrainedModelAdapter:
    def get_recommendations(self, user_id, k, session_items=None, **params):
        ...
        return item_ids[:k]
```

Recommended baseline families (defined as reference implementations in
`examples/reference_recommenders.py`, not shipped by the library):

| Family | Reference class (in examples) | Canonical citation |
|--------|-------------------------------|--------------------|
| Item-item CF | `ItemKNNRecommender` | Deshpande & Karypis, ACM TOIS 2004 |
| Shallow linear CF | `EASERecommender` | Steck, WWW 2019 |
| Sequential CF | `SequentialCFRecommender` | — |

## Runtime params

Reserved keys:

| Key | Purpose |
|-----|---------|
| `num_recs` / `k` | Passed as `k` |
| `layout` | `rows` \| `grid` \| `cards` |
| `swipes_per_refresh` | Swipes before the `cards` deck refreshes its queue |

YAML widget types: `slider`, `selectbox`.

## Layouts

All layouts render ids through the same poster-button card.

| Layout | Behaviour |
|--------|-----------|
| `rows` | Single horizontal row with fixed-width clickable cards and side scroll |
| `grid` | Wrapped clickable poster grid for catalog-style browsing |
| `cards` | Swipe deck: one card at a time with Like / Dislike / Skip; refreshes after `swipes_per_refresh` swipes |
| compare | Forced to `rows` |

Widget keys include rank: `streamlit_recommenders.item.{section}.{id}.r{rank}`. Duplicate ids are deduplicated before display; selected ids remain visible as greyed cards that can be clicked again to unselect.

## Session flow

```mermaid
sequenceDiagram
  participant User
  participant Card
  participant State
  participant Rec as get_recommendations()

  User->>Card: click poster (or Like / Dislike on a swipe card)
  Card->>State: record_selection / record_swipe
  Note over State: selected_ids, disliked_ids, selections updated
  User->>Rec: click Get Recommendations
  Rec->>State: receives session_items (+ history) and selections
  Note over Rec: profile = historical interactions ∪ session likes
  Rec-->>User: refreshed item ids
```

The model scores against the **user profile**, which is the union of the selected user's
historical interactions (`history_item_ids`) and the current session's likes (`session_items`);
`effective_seen` unions the two for seen-filtering. The "Try yourself" session user has no
history, so it is driven purely by session clicks. See the README "How recommendations are
built" section for the end-to-end walk-through.

Reader API in `body()`: `sr.selected_items()`, `sr.disliked_items()`, `sr.displayed_items(label)`, `sr.selections()`, `sr.current_user()`, `sr.param_value(name)`.

`sr.displayed_items(label)` returns the exact ids currently shown in a recommender row (or a
`{label: ids}` dict when called with no argument). It is the single source of truth for what the
UI displays — including the cold-start sample shown for the "Try yourself" session user — so
diagnostics built in `body()` (overlap heatmaps, score tables) stay consistent with the rows on
screen instead of recomputing a different ranking.

## Metrics

`sr.evaluate()` expects recommendations by user and held-out interactions:

```python
sr.evaluate(
    {"Model": {0: [1, 2, 3]}},
    test_interactions=test,
    k=10,
    all_item_ids=items.item_id,
)
```

Implemented: hit rate, recall, NDCG, MRR, coverage.

## Checklist

- [ ] ids are consistent across items/users/interactions/test
- [ ] `get_recommendations(user_id, k, **params)` returns ids from `items`
- [ ] model accepts `session_items` if it should react to card clicks
- [ ] params names match model arguments
