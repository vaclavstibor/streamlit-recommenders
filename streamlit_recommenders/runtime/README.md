# Streamlit runtime internals

Researchers do not import these modules. Demo scripts only call `sr.*`.

## Cache strategy

| What | Mechanism | Invalidation |
|------|-----------|--------------|
| `load_items` / `load_interactions` (file path) | `@st.cache_data` in `cache.load_csv` | File change on disk (Streamlit path hash) |
| `recommend()` output | `@st.cache_data` in `cache.cached_recommend` | Change in `user_id`, `k`, or `hash_params(**params)` |
| DataFrame passed directly to `sr.run()` | No extra cache — already in script memory | — |

## Session state

| Key | Purpose |
|-----|---------|
| `current_user` | Active user for recommendations |
| `clicked_items` | Clicked items (foundation for post-MVP history) |
| `params_snapshot` | Reserved for param diff between reruns |

## Rerun model

Streamlit reruns the entire script on interaction. `sr.run()`:

1. Resolves sidebar params (sliders / selectboxes).
2. Reads `user_id` from the selectbox.
3. Calls `get_recommendations()` — cache hit if params unchanged.
4. Renders the layout.

Data from top-level `pd.read_csv` in a demo script reloads on every rerun — for large data use `sr.load_items(path)`.
