from __future__ import annotations

import re

import streamlit as st

from streamlit_recommenders.layouts._helpers import items_for_recs
from streamlit_recommenders.layouts.item_card import render_horizontal_posters


def _profile_section_key(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "items"


def render_profile_strip(
    items,
    item_ids: list,
    title: str,
    *,
    columns: dict[str, str] | None = None,
    empty_hint: str | None = None,
) -> None:
    entries = items_for_recs(items, item_ids, columns)
    with st.container(border=True):
        st.markdown(f"**{title}**")
        if not entries:
            st.caption(empty_hint or "Nothing here yet.")
            return
        render_horizontal_posters(
            entries,
            section=_profile_section_key(title),
            all_sections=[],
            selectable=False,
        )
