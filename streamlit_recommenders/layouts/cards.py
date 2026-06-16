import streamlit as st

from streamlit_recommenders.layouts._helpers import item_placeholder, items_for_recs
from streamlit_recommenders.runtime.state import record_click


def cards(
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
    for entry in entries:
        with st.container(border=True):
            c1, c2 = st.columns([1, 2.2])
            with c1:
                st.image(entry["image"] or item_placeholder(), width="stretch")
            with c2:
                st.markdown(f"**{entry['title']}**")
                if entry["description"]:
                    st.write(entry["description"])
                if st.button("Add to session", key=f"sr_card_{entry['id']}"):
                    record_click(entry["id"])
