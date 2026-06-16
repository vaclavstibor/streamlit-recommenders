import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st

pio.templates.default = "plotly_white"


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

    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")
