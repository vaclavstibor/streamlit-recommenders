from streamlit_recommenders.layouts.cards import cards
from streamlit_recommenders.layouts.grid import grid
from streamlit_recommenders.layouts.rows import rows

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
    columns: dict[str, str] | None = None,
) -> None:
    fn = LAYOUTS.get(layout, rows)
    fn(items, rec_ids, title=title, columns=columns)
