"""Core rendering for recommendation sections and the swipe deck.

Dispatches the ``rows``, ``grid``, and ``cards`` layouts and wires their
buttons to interaction state (selections, swipes, skips).
"""

from collections.abc import Callable

import streamlit as st

from streamlit_recommenders.layouts._helpers import visible_entries
from streamlit_recommenders.layouts.item_card import (
    DEFAULT_GRID_COLS,
    inject_swipe_styles,
    render_grid_posters,
    render_horizontal_posters,
    render_swipe_card,
    swipe_deck_key,
)
from streamlit_recommenders.runtime.keys import get_recommendations_button_key, section_id
from streamlit_recommenders.runtime.state import (
    bump_swipe_count,
    get_disliked_ids,
    get_selected_ids,
    get_swipe_skipped,
    record_skip,
    record_swipe,
    reset_swipe_count,
)

DEFAULT_SWIPES_PER_REFRESH = 5


def render_item_carousel(
    entries: list[dict],
    section: str,
    all_sections: list[str],
    layout: str,
    *,
    n_cols: int = DEFAULT_GRID_COLS,
) -> None:
    """Draw the shared poster-card carousel for rows and grid layouts.

    Args:
        entries: Prepared item entries to render.
        section: Stable section identifier for state keys.
        all_sections: Every section id on the page.
        layout: Either ``"grid"`` (wrapped gallery) or ``"rows"`` (single
            horizontal strip).
        n_cols: Column count for the grid layout.
    """
    if not entries:
        return

    if layout == "grid":
        render_grid_posters(entries, section, all_sections, n_cols=n_cols)
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
    swipes_per_refresh: int = DEFAULT_SWIPES_PER_REFRESH,
    notice: dict | None = None,
) -> None:
    """Render a titled, bordered recommendation section for any layout.

    Prepares the visible entries, then draws the swipe deck for ``cards`` or a
    poster carousel for ``rows``/``grid``, adding a "Get Recommendations"
    button when a callback is supplied.

    Args:
        layout: One of ``"rows"``, ``"grid"``, or ``"cards"``.
        items: DataFrame of item metadata to draw from.
        rec_ids: Ordered item ids to display.
        title: Heading shown above the section.
        section: Stable section identifier for state keys.
        columns: Optional overrides mapping logical fields to DataFrame
            column names.
        selected_ids: Ids already added to the session profile.
        all_sections: Every section id on the page; defaults to ``[section]``.
        on_get_recommendations: Callback for the action button and swipe
            auto-refresh; the button is omitted when ``None``.
        n_cols: Column count for the grid layout.
        swipes_per_refresh: Swipes before the cards deck auto-refreshes.
        notice: Optional provenance notice (``{"level", "text"}``) shown above
            the items; ``"seed"`` marks a cold-start catalog sample and
            ``"fallback"`` a popularity fallback, so neither is mistaken for the
            model's own output.
    """
    selected = selected_ids or set()
    sections = all_sections or [section]
    entries = visible_entries(items, rec_ids, selected, columns)

    st.subheader(title)
    with st.container(border=True):
        if notice:
            if notice.get("level") == "fallback":
                st.warning(notice["text"], icon="⚠️")
            else:
                st.info(notice["text"], icon="🌱")
        if layout == "cards":
            render_swipe_deck(
                entries,
                section,
                sections,
                on_get_recommendations=on_get_recommendations,
                swipes_per_refresh=swipes_per_refresh,
            )
            return

        if not entries:
            st.info("No recommendations to display.")
        else:
            render_item_carousel(entries, section, sections, layout, n_cols=n_cols)

        if on_get_recommendations is not None:
            st.button(
                "Get Recommendations",
                key=get_recommendations_button_key(section),
                use_container_width=True,
                type="primary",
                on_click=on_get_recommendations,
            )


def render_swipe_deck(
    entries: list[dict],
    section: str,
    all_sections: list[str],
    *,
    on_get_recommendations: Callable[[], None] | None = None,
    swipes_per_refresh: int = DEFAULT_SWIPES_PER_REFRESH,
) -> None:
    """Draw a one-card-at-a-time like/dislike deck with auto-refresh.

    Shows the first unconsumed entry (not already selected, disliked, or
    skipped) with Dislike/Like/Skip buttons. After ``swipes_per_refresh``
    interactions the deck calls ``on_get_recommendations`` to reload.

    Args:
        entries: Prepared item entries queued for the deck.
        section: Stable section identifier for state keys.
        all_sections: Every section id on the page.
        on_get_recommendations: Callback to fetch fresh cards on
            auto-refresh and via the empty-queue button.
        swipes_per_refresh: Swipes before auto-refresh triggers.
    """
    consumed = set(get_selected_ids()) | set(get_disliked_ids()) | set(get_swipe_skipped(section))
    remaining = [entry for entry in entries if entry["id"] not in consumed]

    if not remaining:
        st.info("No more cards in the queue.")
        if on_get_recommendations is not None:
            st.button(
                "Get more recommendations",
                key=f"{_swipe_key(section)}.more",
                use_container_width=True,
                type="primary",
                on_click=on_get_recommendations,
            )
        return

    entry = remaining[0]
    slug = section_id(section)
    deck_key = swipe_deck_key(slug)
    dislike_key = f"{_swipe_key(section)}.dislike.{entry['id']}"
    skip_key = f"{_swipe_key(section)}.skip.{entry['id']}"
    like_key = f"{_swipe_key(section)}.like.{entry['id']}"
    inject_swipe_styles(deck_key, dislike_key, skip_key, like_key)

    with st.container(key=deck_key):
        render_swipe_card(entry)

        dislike_col, like_col = st.columns(2)
        dislike_col.button(
            "Dislike",
            key=dislike_key,
            use_container_width=True,
            on_click=_handle_swipe,
            args=(section, entry["id"], "dislike", all_sections, on_get_recommendations, swipes_per_refresh),
        )
        like_col.button(
            "Like",
            key=like_key,
            use_container_width=True,
            on_click=_handle_swipe,
            args=(section, entry["id"], "like", all_sections, on_get_recommendations, swipes_per_refresh),
        )
        st.button(
            "Skip",
            key=skip_key,
            use_container_width=True,
            on_click=_handle_skip,
            args=(section, entry["id"], on_get_recommendations, swipes_per_refresh),
        )


def _handle_swipe(
    section: str,
    item_id: str | int,
    sentiment: str,
    all_sections: list[str],
    on_get_recommendations: Callable[[], None] | None,
    swipes_per_refresh: int,
) -> None:
    """Record a like/dislike swipe and auto-refresh once the threshold is hit."""
    record_swipe(section, item_id, sentiment, all_sections)
    count = bump_swipe_count(section)
    if count >= swipes_per_refresh and on_get_recommendations is not None:
        on_get_recommendations()
        reset_swipe_count(section)
        st.toast("New recommendations displayed")


def _handle_skip(
    section: str,
    item_id: str | int,
    on_get_recommendations: Callable[[], None] | None,
    swipes_per_refresh: int,
) -> None:
    """Record a skip and auto-refresh once the swipe threshold is hit."""
    record_skip(section, item_id)
    count = bump_swipe_count(section)
    if count >= swipes_per_refresh and on_get_recommendations is not None:
        on_get_recommendations()
        reset_swipe_count(section)
        st.toast("New recommendations displayed")


def _swipe_key(section: str) -> str:
    """Return the namespaced state-key prefix for a section's swipe widgets."""
    return f"streamlit_recommenders.swipe.{section_id(section)}"
