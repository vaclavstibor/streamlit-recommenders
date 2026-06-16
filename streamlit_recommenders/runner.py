from collections.abc import Callable
from typing import Any

import pandas as pd
import streamlit as st

from streamlit_recommenders.config.yaml_loader import load_config
from streamlit_recommenders.layouts import render_layout
from streamlit_recommenders.models.adapter import adapt_recommender
from streamlit_recommenders.runtime.cache import get_recommendations, load_csv
from streamlit_recommenders.runtime.state import get_clicked_items, init_session_state, set_current_user
from streamlit_recommenders.widgets.params import resolve_params


def load_items(path: str, id_col: str = "item_id") -> pd.DataFrame:
    df = load_csv(path)
    if id_col not in df.columns:
        raise ValueError(f"Column '{id_col}' not found in {path}")
    return df


def load_interactions(path: str) -> pd.DataFrame:
    return load_csv(path)


def run(
    recommend: Callable[..., list],
    items: pd.DataFrame,
    interactions: pd.DataFrame | None = None,
    layout: str = "rows",
    params: dict[str, Any] | None = None,
    config: str | None = None,
    title: str = "Recommender Demo",
    subtitle: str | None = None,
    item_columns: dict[str, str] | None = None,
    body: Callable[[], None] | None = None,
) -> None:
    """Orchestrate the full Streamlit demo app."""
    st.set_page_config(page_title=title, layout="wide")
    init_session_state()

    cfg = load_config(config)
    layout = cfg.get("layout", layout)
    item_columns = cfg.get("item_columns", item_columns)
    yaml_params = cfg.get("params")
    subtitle = subtitle or cfg.get("subtitle")

    st.title(title)
    if subtitle:
        st.caption(subtitle)

    st.sidebar.subheader("User")
    user_id = _select_user(interactions, items)
    set_current_user(user_id)

    st.sidebar.subheader("Parameters")
    resolved = resolve_params(params, yaml_params)
    k = int(resolved.pop("num_recs", resolved.pop("k", 10)))
    layout = resolved.pop("layout", layout)

    recommend_fn = adapt_recommender(recommend)
    rec_ids = get_recommendations(recommend_fn, user_id, k, resolved)

    render_layout(layout, items, rec_ids, columns=item_columns)

    clicked = get_clicked_items()
    if clicked:
        st.caption(f"Saved this session: {', '.join(str(i) for i in clicked)}")

    if body:
        with st.container(border=True):
            body()


def _select_user(
    interactions: pd.DataFrame | None,
    items: pd.DataFrame,
) -> str | int:
    if interactions is not None and "user_id" in interactions.columns:
        users = sorted(interactions["user_id"].unique())
        return st.sidebar.selectbox("User", users, key="sr_user_select", label_visibility="collapsed")

    id_col = "item_id" if "item_id" in items.columns else items.columns[0]
    return st.sidebar.selectbox(
        "Context",
        items[id_col].head(20).tolist(),
        key="sr_user_select",
        label_visibility="collapsed",
    )
