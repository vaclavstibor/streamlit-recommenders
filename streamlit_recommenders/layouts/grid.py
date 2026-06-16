import streamlit as st

from streamlit_recommenders.layouts._helpers import item_placeholder, items_for_recs
from streamlit_recommenders.runtime.state import record_click


def grid(
    items,
    rec_ids: list,
    title: str = "Recommended for you",
    columns: dict[str, str] | None = None,
    n_cols: int = 4,
) -> None:
    entries = items_for_recs(items, rec_ids, columns)
    if not entries:
        st.info("No recommendations to display.")
        return

    st.subheader(title)
    n_cols = max(1, min(n_cols, 4))
    with st.container(border=True):
        for start in range(0, len(entries), n_cols):
            chunk = entries[start : start + n_cols]
            cols = st.columns(n_cols)
            for col, entry in zip(cols, chunk):
                with col:
                    st.image(entry["image"] or item_placeholder(), width="stretch")
                    st.markdown(f"**{entry['title']}**")
                    if entry["description"]:
                        st.caption(entry["description"][:90])
                    if st.button("View", key=f"sr_grid_{entry['id']}", use_container_width=True):
                        record_click(entry["id"])
