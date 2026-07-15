"""Plotly-backed chart helpers rendered into Streamlit."""

import pandas as pd
import plotly.express as px
import streamlit as st


def _style(fig, **overrides) -> None:
    """Apply the shared chart layout to ``fig``, overriding with ``overrides``."""
    layout = {
        "template": "plotly_white",
        "margin": dict(l=20, r=20, t=40, b=20),
        "paper_bgcolor": "rgba(0,0,0,0)",
    }
    layout.update(overrides)
    fig.update_layout(**layout)


def plot(
    df: pd.DataFrame,
    x: str | None = None,
    y: str | None = None,
    color: str | None = None,
    title: str | None = None,
) -> None:
    """Render a chart chosen from ``df`` and column types.

    Picks a scatter when ``color`` is set, a line for numeric ``y``, else a bar.
    Missing ``x``/``y`` default to the first column and first numeric column.

    Args:
        df: Data to plot.
        x: Column for the x axis; defaults to the first column.
        y: Column for the y axis; defaults to the first numeric column.
        color: Column used to color a scatter plot.
        title: Chart title.
    """
    if df.empty:
        st.info("Nothing to plot.")
        return

    if x is None:
        x = df.columns[0]
    if y is None:
        numeric = df.select_dtypes(include="number").columns
        y = numeric[0] if len(numeric) else df.columns[-1]

    if color:
        fig = px.scatter(df, x=x, y=y, color=color, title=title)
    else:
        fig = (
            px.line(df, x=x, y=y, title=title)
            if df[y].dtype.kind in "ifc"
            else px.bar(df, x=x, y=y, title=title)
        )

    _style(fig)
    st.plotly_chart(fig, width="stretch")


def plot_metric_comparison(
    df: pd.DataFrame,
    *,
    metric_col: str = "metric",
    value_col: str = "value",
    model_col: str = "model",
    title: str = "Metric comparison",
) -> None:
    """Render a grouped bar chart comparing metric values across models.

    Args:
        df: Long-form metrics table.
        metric_col: Column naming each metric (x axis).
        value_col: Column holding metric values (y axis).
        model_col: Column identifying each model (bar color).
        title: Chart title.
    """
    if df.empty:
        st.info("No metrics to plot.")
        return
    fig = px.bar(
        df,
        x=metric_col,
        y=value_col,
        color=model_col,
        barmode="group",
        title=title,
    )
    _style(fig)
    st.plotly_chart(fig, width="stretch")


def plot_ranked_items(
    df: pd.DataFrame,
    *,
    title_col: str = "title",
    score_col: str = "score",
    title: str = "Ranked items",
) -> None:
    """Render a bar chart of item scores.

    Args:
        df: Ranked items table.
        title_col: Column holding item titles (x axis).
        score_col: Column holding scores (y axis).
        title: Chart title.
    """
    if df.empty:
        st.info("No ranked items to plot.")
        return
    fig = px.bar(df, x=title_col, y=score_col, title=title)
    _style(fig)
    st.plotly_chart(fig, width="stretch")


def plot_score_distribution(
    df: pd.DataFrame,
    *,
    score_col: str = "score",
    title: str = "Score distribution",
) -> None:
    """Render a histogram of recommendation scores.

    Args:
        df: Table containing scores.
        score_col: Column holding the scores to bin.
        title: Chart title.
    """
    if df.empty:
        st.info("No scores to plot.")
        return
    fig = px.histogram(df, x=score_col, title=title)
    _style(fig)
    st.plotly_chart(fig, width="stretch")


def recommendation_overlap_matrix(recommendations: dict[str, list]) -> pd.DataFrame:
    """Compute a pairwise Jaccard overlap matrix between recommendation lists.

    Args:
        recommendations: Mapping of model name to its recommended item ids.

    Returns:
        A square DataFrame indexed and columned by model name whose cells hold
        the Jaccard similarity of each model pair (0.0 when both are empty).
    """
    names = list(recommendations)
    matrix = pd.DataFrame(1.0, index=names, columns=names)
    for left in names:
        for right in names:
            left_set = set(recommendations[left])
            right_set = set(recommendations[right])
            union = left_set | right_set
            matrix.loc[left, right] = len(left_set & right_set) / len(union) if union else 0.0
    return matrix


def plot_overlap_heatmap(
    overlap: pd.DataFrame,
    *,
    title: str = "Recommendation overlap",
) -> None:
    """Render an overlap matrix as an annotated heatmap.

    Args:
        overlap: Square overlap matrix, e.g. from
            :func:`recommendation_overlap_matrix`.
        title: Chart title.
    """
    if overlap.empty:
        st.info("No recommendation overlap to plot.")
        return
    fig = px.imshow(
        overlap,
        text_auto=".2f",
        zmin=0,
        zmax=1,
        color_continuous_scale="Blues",
        title=title,
    )
    _style(
        fig,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="Model",
        yaxis_title="Model",
    )
    st.plotly_chart(fig, width="stretch")
