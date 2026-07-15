"""Grid layout: a wrapped gallery of selectable poster cards."""

from streamlit_recommenders.layouts.section import render_recommender_section


def grid(
    items,
    rec_ids: list,
    title: str = "Recommended for you",
    section: str = "default",
    columns: dict[str, str] | None = None,
    n_cols: int = 10,
    selected_ids: set[str | int] | None = None,
    all_sections: list[str] | None = None,
) -> None:
    """Render recommendations as a wrapped grid of poster cards.

    Args:
        items: DataFrame of item metadata to draw from.
        rec_ids: Ordered item ids to display.
        title: Heading shown above the grid.
        section: Stable section identifier for state keys.
        columns: Optional overrides mapping logical fields to DataFrame
            column names.
        n_cols: Number of poster columns per row.
        selected_ids: Ids already added to the session profile.
        all_sections: Every section id on the page, used when recording
            selections across sections.
    """
    render_recommender_section(
        "grid",
        items,
        rec_ids,
        title=title,
        section=section,
        columns=columns,
        selected_ids=selected_ids,
        all_sections=all_sections,
        n_cols=n_cols,
    )
