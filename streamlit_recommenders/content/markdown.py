from pathlib import Path

import streamlit as st


def markdown(text: str) -> None:
    st.markdown(text)


def markdown_file(path: str | Path) -> None:
    with open(path) as f:
        st.markdown(f.read())
