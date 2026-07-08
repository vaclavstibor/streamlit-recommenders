# Runtime internals

Not part of the public API. Demo scripts use `sr.*` only.

## Cache

| What | Where |
|------|-------|
| CSV load | `@st.cache_data` in `cache.load_csv` |
| `get_recommendations()` | `@st.cache_data` keyed by user, k, params, session_items, selections |

## Session state

| Key | Purpose |
|-----|---------|
| `current_user` | Sidebar selection |
| `selected_ids` | Items clicked this session |
| `selections` | Per-section `{item_id, rank, source}` |
| `displayed_recs` | Carousel ids per recommender section |
| `run_context_hash` | Clears displayed recs on user/k/params change |

## Rerun flow (`runner.py`)

1. Resolve sidebar params and user
2. Load or reuse `displayed_recs` per section
3. Render profile strips + item carousel
4. **Get Recommendations** button recomputes and stores new ids, then `st.rerun()`

Card click → `record_selection()` → immediate rerun. The selected item stays in the carousel as a greyed card, appears in the profile strip, and can be clicked again to unselect.
