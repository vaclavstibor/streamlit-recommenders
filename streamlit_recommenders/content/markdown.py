"""Helpers for rendering Markdown content into Streamlit."""

from pathlib import Path

import streamlit as st


def markdown(text: str) -> None:
    """Render a Markdown string.

    Args:
        text: Markdown source to render.
    """
    st.markdown(text)


def markdown_file(path: str | Path) -> None:
    """Read a file and render its contents as Markdown.

    Args:
        path: Path to the Markdown file.
    """
    with open(path) as f:
        st.markdown(f.read())
