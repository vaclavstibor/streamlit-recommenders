import pandas as pd
import streamlit as st


def table(df: pd.DataFrame, height: int | str | None = None) -> None:
    kwargs: dict = {"width": "stretch"}
    if height is not None:
        kwargs["height"] = height
    st.dataframe(df, **kwargs)
