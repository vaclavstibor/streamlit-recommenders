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
