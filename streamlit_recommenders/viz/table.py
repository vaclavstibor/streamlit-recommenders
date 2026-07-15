"""Streamlit dataframe rendering helper."""

import pandas as pd
import streamlit as st


def table(df: pd.DataFrame, height: int | str | None = None) -> None:
    """Render ``df`` as a full-width Streamlit dataframe.

    Args:
        df: Data to display.
        height: Optional fixed table height in pixels; omitted when ``None``.
    """
    kwargs: dict = {"width": "stretch"}
    if height is not None:
        kwargs["height"] = height
    st.dataframe(df, **kwargs)
