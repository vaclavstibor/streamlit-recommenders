# Runtime internals

Not public API; demos use `sr.*`.

## Cache

| What | How |
|------|-----|
| CSV loads | `st.cache_data` in `cache.load_csv` |
| `get_recommendations()` | `st.cache_data`, keyed by model identity, user, k, params, session items and selections |

## Session state (`st.session_state["_sr_state"]`)

| Key | Content |
|-----|---------|
| `current_user` | sidebar selection; changing it resets everything below |
| `selected_ids` | liked/clicked items (the positive session profile) |
| `disliked_ids` | swipe dislikes |
| `selections` | per-section `{item_id, rank, source[, sentiment]}` entries |
| `displayed_recs` | ids currently shown, per section |
| `displayed_notice` | seed/fallback label per section |
| `swipe_counts`, `swipe_skipped` | cards deck progress |
| `run_context_hash` | user + k + params; a change clears displayed recs and deck progress |
| `cold_start_seed` | per-session seed for the cold-start sample |

## One rerun (`runner.py`)

1. Resolve the user and sidebar params; reset state if the run context changed.
2. Render profile strips (history, selected, disliked).
3. For each section, reuse `displayed_recs` or compute them (cold-start sample or cached model call).
4. Render the layout. Buttons use `on_click` callbacks, so a click updates state before the next rerun.

A card click toggles the item in `selected_ids`; the card shows a *Selected* badge and the item
appears in the *Selected this session* strip, where clicking it again removes it.
