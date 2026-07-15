"""Horizontal strip widget for displaying a user's profile items."""

from __future__ import annotations

import re

import streamlit as st

from streamlit_recommenders.layouts._helpers import items_for_recs
from streamlit_recommenders.layouts.item_card import render_horizontal_posters


def _profile_section_key(title: str) -> str:
    """Slugify ``title`` into a section key, falling back to ``"items"``."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "items"


def render_profile_strip(
    items,
    item_ids: list,
    title: str,
    *,
    columns: dict[str, str] | None = None,
    empty_hint: str | None = None,
    selectable: bool = False,
    all_sections: list[str] | None = None,
) -> None:
    """Render a horizontal strip of profile items inside a bordered container.

    ``selectable=True`` makes the cards clickable so items can be unselected
    directly from the strip.

    Args:
        items: Item catalog used to resolve ``item_ids`` into card entries.
        item_ids: Ids of items to show, in order.
        title: Heading shown above the strip.
        columns: Optional column-name mapping passed to card resolution.
        empty_hint: Caption shown when there are no items.
        selectable: Whether cards render as clickable/unselectable.
        all_sections: Section keys for cross-section selection state.
    """
    entries = items_for_recs(items, item_ids, columns)
    if selectable:
        for entry in entries:
            entry["selected"] = True
    with st.container(border=True):
        st.markdown(f"**{title}**")
        if not entries:
            st.caption(empty_hint or "Nothing here yet.")
            return
        render_horizontal_posters(
            entries,
            section=_profile_section_key(title),
            all_sections=all_sections or [],
            selectable=selectable,
        )
