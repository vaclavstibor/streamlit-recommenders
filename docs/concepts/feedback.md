# Feedback & session

This page is the precise answer to "what does the framework do with my feedback, and what does my
model have to do?" — because the boundary matters for interpreting what you see.

## The user profile

A model always scores against a **user profile**: the union of the selected user's **history**
(their rows in `interactions`) and the **positive** interactions from the current session. The
"Try yourself" session user has no history, so it is driven purely by what you click.

## Feedback signals

The three click/swipe actions map onto the contract as follows. The framework only **passes** them
— what a model does with each is its own decision:

| Action | Carried as | What the framework does | What the built-in models do | What a custom model can do |
|--------|-----------|-------------------------|-----------------------------|----------------------------|
| **Like / click** | `session_items` (+ a `selections` entry) | Adds to the positive profile | Fold into the profile vector | Same |
| **Dislike** (cards) | `selections` entry with `sentiment: "dislike"` | Passes it through; records it in the *Disliked this session* strip | **Exclude the item only** (no down-weighting) | Read the sentiment and treat it as a negative signal |
| **Skip** (cards) | Excluded ids (not in `selections`) | Marks as seen | Excluded so the card does not reappear | — |

!!! note "Dislikes are an available signal, not an automatic one"
    The `BaseRecommender` / `ArtifactRecommender` baselines **exclude** a disliked item but do not
    use it to push down similar items. Because `selections` is passed through untouched, a model
    that overrides `get_recommendations(..., selections=...)` can read the dislike sentiment and act
    on it — see
    [`examples/swipe_deck_cards.py::FeedbackAwareEASE`](https://github.com/vaclavstibor/streamlit-recommenders/blob/main/examples/swipe_deck_cards.py).

Refresh happens on **Get Recommendations** (rows/grid) or automatically after `swipes_per_refresh`
swipes (cards). Reader API for `body()`: `sr.selected_items()`, `sr.disliked_items()`,
`sr.displayed_items(label)`, `sr.selections()`, `sr.current_user()`, `sr.param_value(name)`.

## Seen-filtering — and how to turn it off

By default, items already in the user's history or current session are filtered out of
recommendations (`effective_seen`). To inspect a model's **raw ranking including seen items** —
useful for checking its unfiltered static output — turn off the **Hide already-seen items** sidebar
toggle. The default (on) keeps the usual "don't re-recommend what you've seen" behavior.

The toggle applies to the built-in `BaseRecommender`/`ArtifactRecommender` ranking path; a plain
function does its own filtering.

## No silent fallbacks

An inspection tool must never present something that is not the model's own output as if it were.
Two cases are surfaced explicitly:

- **Empty session profile (cold start).** Before you click anything, the "Try yourself" user has no
  signal. Rather than fabricate a ranking, the app shows a clearly labeled **catalog seed sample**
  (🌱 *"catalog sample to seed the profile, not model recommendations"*) for you to react to. The
  model is not called until the profile has something in it.
- **Popularity fallback.** If a loaded artifact genuinely has no signal for a profile (e.g. ids
  outside its catalog, or a sequential anchor with no transitions), it falls back to global
  popularity — and the app flags this with a ⚠️ **"Popularity fallback"** badge above the row, so
  the result is never misattributed to the model. `ArtifactRecommender.fallback_reason()` reports
  when this happens.

Because a demo user with real history and a mapped catalog never hits either case, both are edge
states — but they are shown, not hidden.
