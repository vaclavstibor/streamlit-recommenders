"""Layout registry and dispatch for rendering recommendation sections."""

from collections.abc import Callable

from streamlit_recommenders.layouts.cards import cards
from streamlit_recommenders.layouts.grid import grid
from streamlit_recommenders.layouts.rows import rows
from streamlit_recommenders.layouts.section import DEFAULT_SWIPES_PER_REFRESH, render_recommender_section

LAYOUTS = {
    "rows": rows,
    "grid": grid,
    "cards": cards,
}


def render_layout(
    layout: str,
    items,
    rec_ids: list,
    title: str = "Recommendations",
    section: str | None = None,
    columns: dict[str, str] | None = None,
    selected_ids: set[str | int] | None = None,
    all_sections: list[str] | None = None,
    on_get_recommendations: Callable[[], None] | None = None,
    n_cols: int = 10,
    swipes_per_refresh: int = DEFAULT_SWIPES_PER_REFRESH,
    notice: dict | None = None,
) -> None:
    """Render a recommendation section using the named layout.

    Args:
        layout: Layout name; one of ``"rows"``, ``"grid"``, or ``"cards"``.
        items: DataFrame of item metadata to draw from.
        rec_ids: Ordered item ids to display.
        title: Heading shown above the section.
        section: Stable section identifier for state keys; defaults to
            ``title`` when omitted.
        columns: Optional overrides mapping logical fields to DataFrame
            column names.
        selected_ids: Ids already added to the session profile.
        all_sections: Every section id on the page, used when recording
            interactions across sections.
        on_get_recommendations: Callback invoked by the section's action
            button (or swipe auto-refresh).
        n_cols: Column count for the grid layout.
        swipes_per_refresh: Swipes before the cards deck auto-refreshes.
        notice: Optional provenance notice (``{"level", "text"}``) shown above
            the items when they are a seed sample or popularity fallback rather
            than the model's own output.
    """
    render_recommender_section(
        layout,
        items,
        rec_ids,
        title=title,
        section=section or title,
        columns=columns,
        selected_ids=selected_ids,
        all_sections=all_sections,
        on_get_recommendations=on_get_recommendations,
        n_cols=n_cols,
        swipes_per_refresh=swipes_per_refresh,
        notice=notice,
    )
