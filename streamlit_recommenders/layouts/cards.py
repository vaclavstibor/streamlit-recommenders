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
    """Swipe deck: one card at a time, Like/Dislike/Skip, auto-refresh after N swipes."""
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
