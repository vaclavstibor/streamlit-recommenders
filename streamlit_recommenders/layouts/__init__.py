from collections.abc import Callable

from streamlit_recommenders.layouts.cards import cards
from streamlit_recommenders.layouts.grid import grid
from streamlit_recommenders.layouts.rows import rows
from streamlit_recommenders.layouts.section import render_recommender_section

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
    n_cols: int = 4,
) -> None:
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
    )
