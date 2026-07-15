"""Cards layout: a Tinder-style swipe deck of recommendations."""

from streamlit_recommenders.layouts.section import (
    DEFAULT_SWIPES_PER_REFRESH,
    render_recommender_section,
)


def cards(
    items,
    rec_ids: list,
    title: str = "Recommended for you",
    section: str = "default",
    columns: dict[str, str] | None = None,
    selected_ids: set[str | int] | None = None,
    all_sections: list[str] | None = None,
    on_get_recommendations=None,
    swipes_per_refresh: int = DEFAULT_SWIPES_PER_REFRESH,
) -> None:
    """Swipe deck: one card at a time, Like/Dislike/Skip, auto-refresh after N swipes.

    Args:
        items: DataFrame of item metadata to draw from.
        rec_ids: Ordered item ids to queue into the deck.
        title: Heading shown above the deck.
        section: Stable section identifier for state keys.
        columns: Optional overrides mapping logical fields to DataFrame
            column names.
        selected_ids: Ids already added to the session profile.
        all_sections: Every section id on the page, used when recording
            swipes across sections.
        on_get_recommendations: Callback fired to fetch fresh cards, both on
            auto-refresh and via the "Get more recommendations" button.
        swipes_per_refresh: Number of swipes before auto-refresh triggers.
    """
    render_recommender_section(
        "cards",
        items,
        rec_ids,
        title=title,
        section=section,
        columns=columns,
        selected_ids=selected_ids,
        all_sections=all_sections,
        on_get_recommendations=on_get_recommendations,
        swipes_per_refresh=swipes_per_refresh,
    )
