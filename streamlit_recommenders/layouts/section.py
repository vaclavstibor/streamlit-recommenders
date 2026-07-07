from collections.abc import Callable

import streamlit as st

from streamlit_recommenders.layouts._helpers import visible_entries
from streamlit_recommenders.layouts.item_card import (
    DEFAULT_GRID_COLS,
    render_grid_posters,
    render_horizontal_posters,
)
from streamlit_recommenders.runtime.keys import get_recommendations_button_key


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
        render_grid_posters(entries, section, all_sections, n_cols=n_cols)
        return

    if layout == "cards":
        render_grid_posters(entries, section, all_sections, n_cols=max(n_cols, 5))
        return

    # rows: one horizontally scrollable line for model comparison
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
    on_get_recommendations: Callable[[], None] | None = None,
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
            if layout in {"grid", "cards"}:
                st.caption("Click posters to add items to the session profile.")
            render_item_carousel(entries, section, sections, layout, n_cols=n_cols)

        if on_get_recommendations is not None:
            st.button(
                "Get Recommendations",
                key=get_recommendations_button_key(section),
                use_container_width=True,
                type="primary",
                on_click=on_get_recommendations,
            )
