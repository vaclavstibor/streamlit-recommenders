from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from streamlit_recommenders.viz.table import table


def dataset_info(
    items: pd.DataFrame,
    interactions: pd.DataFrame,
    *,
    users: pd.DataFrame | None = None,
    title: str = "Dataset info",
    item_id_col: str = "item_id",
    user_id_col: str = "user_id",
    genre_cols: tuple[str, ...] = ("genres", "genre"),
    exclude_item_cols: tuple[str, ...] = ("image_url", "poster_path", "description"),
    expanded: bool = False,
    key_prefix: str = "streamlit_recommenders.dataset_info",
) -> None:
    """Render a compact, reusable dataset inspection block."""
    with st.expander(title, expanded=expanded):
        table(_dataset_stats(items, interactions, users, user_id_col=user_id_col))

        genre_col = next((col for col in genre_cols if col in items.columns), None)
        categorical_cols = [
            col
            for col in items.columns
            if items[col].dtype == "object" and col not in set(exclude_item_cols)
        ]

        plot_options = ["Interactions per user"]
        if genre_col:
            plot_options.append("Item genres")
        if categorical_cols:
            plot_options.append("Item metadata column")

        plot_choice = st.selectbox(
            "Distribution to plot",
            plot_options,
            key=f"{key_prefix}.plot_choice",
        )

        if plot_choice == "Item genres" and genre_col:
            _plot_item_genres(items, item_id_col=item_id_col, genre_col=genre_col)
        elif plot_choice == "Item metadata column" and categorical_cols:
            column = st.selectbox(
                "Item column",
                categorical_cols,
                key=f"{key_prefix}.item_column",
            )
            _plot_item_column(items, column)
        else:
            _plot_interactions_per_user(interactions, user_id_col=user_id_col)


def _dataset_stats(
    items: pd.DataFrame,
    interactions: pd.DataFrame,
    users: pd.DataFrame | None,
    *,
    user_id_col: str,
) -> pd.DataFrame:
    rows = [
        {"table": "items", "rows": len(items), "columns": len(items.columns)},
        {
            "table": "interactions",
            "rows": len(interactions),
            "columns": len(interactions.columns),
        },
    ]
    if user_id_col in interactions.columns:
        rows.append(
            {
                "table": "users in interactions",
                "rows": interactions[user_id_col].nunique(),
                "columns": None,
            }
        )
    if users is not None:
        rows.append({"table": "users", "rows": len(users), "columns": len(users.columns)})
    return pd.DataFrame(rows)


def _plot_item_genres(
    items: pd.DataFrame,
    *,
    item_id_col: str,
    genre_col: str,
) -> None:
    columns = [col for col in [item_id_col, genre_col] if col in items.columns]
    genres = (
        items[columns]
        .dropna(subset=[genre_col])
        .assign(**{genre_col: lambda df: df[genre_col].astype(str).str.split("|")})
        .explode(genre_col)
    )
    genre_counts = genres[genre_col].value_counts().head(15).reset_index()
    genre_counts.columns = ["genre", "items"]
    fig = px.bar(genre_counts, x="genre", y="items", title="Top item genres")
    _show_plot(fig)


def _plot_item_column(items: pd.DataFrame, column: str) -> None:
    counts = items[column].dropna().astype(str).value_counts().head(20).reset_index()
    counts.columns = [column, "items"]
    fig = px.bar(counts, x=column, y="items", title=f"Top values in {column}")
    _show_plot(fig)


def _plot_interactions_per_user(interactions: pd.DataFrame, *, user_id_col: str) -> None:
    if user_id_col not in interactions.columns:
        st.info(f"No `{user_id_col}` column available for interaction distribution.")
        return
    interactions_per_user = interactions.groupby(user_id_col).size().reset_index(name="interactions")
    fig = px.histogram(
        interactions_per_user,
        x="interactions",
        nbins=40,
        title="Interactions per user",
    )
    _show_plot(fig)


def _show_plot(fig) -> None:
    fig.update_layout(margin=dict(l=20, r=20, t=45, b=20), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")
