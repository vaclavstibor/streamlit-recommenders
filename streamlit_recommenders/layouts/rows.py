"""Rows layout: a single horizontally scrollable strip of poster cards."""

from streamlit_recommenders.layouts.section import render_recommender_section


def rows(
    items,
    rec_ids: list,
    title: str = "Recommended for you",
    section: str = "default",
    columns: dict[str, str] | None = None,
    selected_ids: set[str | int] | None = None,
    all_sections: list[str] | None = None,
) -> None:
    """Render recommendations as one horizontally scrollable poster strip.

    Args:
        items: DataFrame of item metadata to draw from.
        rec_ids: Ordered item ids to display.
        title: Heading shown above the strip.
        section: Stable section identifier for state keys.
        columns: Optional overrides mapping logical fields to DataFrame
            column names.
        selected_ids: Ids already added to the session profile.
        all_sections: Every section id on the page, used when recording
            selections across sections.
    """
    render_recommender_section(
        "rows",
        items,
        rec_ids,
        title=title,
        section=section,
        columns=columns,
        selected_ids=selected_ids,
        all_sections=all_sections,
    )
