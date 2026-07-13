import pandas as pd
import plotly.express as px
import streamlit as st


def _style(fig, **overrides) -> None:
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
    if df.empty:
        st.info("No scores to plot.")
        return
    fig = px.histogram(df, x=score_col, title=title)
    _style(fig)
    st.plotly_chart(fig, width="stretch")


def recommendation_overlap_matrix(recommendations: dict[str, list]) -> pd.DataFrame:
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
