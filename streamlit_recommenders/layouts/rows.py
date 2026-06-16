import streamlit as st

from streamlit_recommenders.layouts._helpers import item_placeholder, items_for_recs
from streamlit_recommenders.runtime.state import record_click


def rows(
    items,
    rec_ids: list,
    title: str = "Recommended for you",
    columns: dict[str, str] | None = None,
) -> None:
    entries = items_for_recs(items, rec_ids, columns)
    if not entries:
        st.info("No recommendations to display.")
        return

    st.subheader(title)
    with st.container(border=True):
        cols = st.columns(len(entries))
        for col, entry in zip(cols, entries):
            with col:
                st.image(entry["image"] or item_placeholder(), width=130)
                st.markdown(f"**{entry['title']}**")
                if entry["description"]:
                    st.caption(entry["description"][:72])
                if st.button("Save", key=f"sr_row_{entry['id']}", use_container_width=True):
                    record_click(entry["id"])
