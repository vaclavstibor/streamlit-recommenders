from collections.abc import Callable

import streamlit as st

from streamlit_recommenders.layouts._helpers import visible_entries
from streamlit_recommenders.layouts.item_card import DEFAULT_GRID_COLS, render_horizontal_posters
from streamlit_recommenders.runtime.keys import recommend_button_key


def render_item_carousel(
    entries: list[dict],
    section: str,
    all_sections: list[str],
    layout: str,
    *,
    n_cols: int = DEFAULT_GRID_COLS,
) -> None:
    """Shared poster-card carousel for rows, grid, and cards layouts."""
    if not entries:
        return

    if layout == "grid":
        n_cols = max(1, min(n_cols, len(entries)))
        for start in range(0, len(entries), n_cols):
            render_horizontal_posters(
                entries[start : start + n_cols],
                section,
                all_sections,
                rank_offset=start,
            )
        return

    # rows and cards: one horizontally scrollable line
    render_horizontal_posters(entries, section, all_sections)


def render_recommender_section(
    layout: str,
    items,
    rec_ids: list,
    title: str,
    section: str,
    *,
    columns: dict[str, str] | None = None,
    selected_ids: set[str | int] | None = None,
    all_sections: list[str] | None = None,
    on_recommend: Callable[[], None] | None = None,
    n_cols: int = DEFAULT_GRID_COLS,
) -> None:
    selected = selected_ids or set()
    sections = all_sections or [section]
    entries = visible_entries(items, rec_ids, selected, columns)

    st.subheader(title)
    with st.container(border=True):
        if not entries:
            st.info("No recommendations to display.")
        else:
            render_item_carousel(entries, section, sections, layout, n_cols=n_cols)

        if on_recommend is not None and st.button(
            "Recommend",
            key=recommend_button_key(section),
            use_container_width=True,
            type="primary",
        ):
            on_recommend()
            st.rerun()
