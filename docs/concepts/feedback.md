# Feedback & session

What happens to a click, Like, Dislike or Skip, and what your model receives because of it.

## The user profile

A model scores against a profile: the selected user's history (their rows in `interactions`)
plus the items liked or clicked in the current session. The *Try it yourself* user has no history,
so its profile is only what you pick.

## Feedback signals

| Action | Your model receives | `BaseRecommender` / `ArtifactRecommender` |
|--------|---------------------|-------------------------------------------|
| Click a card / Like | the id in `session_items` (plus a `selections` entry) | adds it to the profile |
| Dislike (cards only) | a `selections` entry with `sentiment: "dislike"` | treats it as seen and passes it to `scores(disliked_items=...)` |
| Skip (cards only) | nothing; the card is filtered out of the deck | — |

Results refresh on **Get Recommendations** (rows, grid) or every `swipes_per_refresh` swipes (cards).

### How a Dislike changes the ranking

`ArtifactRecommender` subtracts `dislike_weight` (default `1.0`) times the disliked item's weight
row from the scores. For item-item models (ItemKNN, EASE) that pushes down items similar to the
rejected one; for sequential CF it pushes down items that usually follow it. With
`dislike_weight=0` a dislike only hides the item. On ml-latest-small with EASE, after liking
*Toy Story* and *A Bug's Life*, disliking *Toy Story 2* drops *The Lion King* and *Beauty and the
Beast* from the top 8.

Dislikes do not form a profile on their own: a session with dislikes but no likes still gets the
popularity fallback, labeled as such (see below).

To use dislikes in your own model, either declare `disliked_items` in a `BaseRecommender.scores()`
override, or read `selections` in a plain function:

```python
def get_recommendations(user_id, k, session_items=None, selections=None, **params):
    disliked = {s["item_id"] for s in selections or [] if s.get("sentiment") == "dislike"}
    ...
```

Expose the weight as a sidebar control with `params={"dislike_weight": sr.slider("Dislike weight", 0.0, 3.0, 1.0)}`.

In `body()` the same state is readable through `sr.selected_items()`, `sr.disliked_items()`,
`sr.displayed_items(label)`, `sr.selections()`, `sr.current_user()` and `sr.param_value(name)`.

## Seen items

By default, items in the user's history or current session (including dislikes) are filtered out.
Turn off **Hide already-seen items** in the sidebar to see the model's raw ranking. The toggle
applies to `BaseRecommender` and `ArtifactRecommender`; a plain function does its own filtering
and receives the flag as `hide_seen` if it declares it.

## Nothing is silently attributed to the model

Two situations would otherwise show items the model did not choose. Both are labeled in the UI:

- **Empty session profile.** Before the *Try it yourself* user picks anything, the app shows a sample
  of popular catalog items, marked as a seed sample. The model is not called until there is
  something in the profile.
- **Popularity fallback.** When an `ArtifactRecommender` has no signal for the profile (ids outside
  its catalog, only dislikes, a sequential anchor without transitions), it returns global
  popularity and the section shows a *Popularity fallback* warning.
  `ArtifactRecommender.fallback_reason()` reports the same thing in code.
