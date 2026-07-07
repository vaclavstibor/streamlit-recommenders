from collections.abc import Callable
from typing import Any

import pandas as pd
import streamlit as st

from streamlit_recommenders.config.yaml_loader import load_config
from streamlit_recommenders.data import load_interactions as _load_interactions
from streamlit_recommenders.data import load_items as _load_items
from streamlit_recommenders.layouts import render_layout
from streamlit_recommenders.runtime.cache import get_recommendations as cached_get_recommendations
from streamlit_recommenders.runtime.compare import (
    GetRecommendationsInput,
    is_compare_mode,
    labeled_get_recommendations,
)
from streamlit_recommenders.runtime.seen import SESSION_USER_ID, SESSION_USER_LABEL
from streamlit_recommenders.runtime.state import (
    get_displayed_recs,
    get_selected_ids,
    get_selections,
    init_session_state,
    set_current_user,
    set_displayed_recs,
    sync_run_context,
)
from streamlit_recommenders.runtime.keys import user_select_key
from streamlit_recommenders.widgets.params import (
    resolve_model_params,
    resolve_params,
    split_model_params,
)
from streamlit_recommenders.widgets.profile_strip import render_profile_strip
from streamlit_recommenders.widgets.user_profile import history_item_ids, render_user_profile


def load_items(path: str, id_col: str = "item_id") -> pd.DataFrame:
    return _load_items(path, id_col)


def load_interactions(
    path: str,
    user_col: str = "user_id",
    item_col: str = "item_id",
) -> pd.DataFrame:
    return _load_interactions(path, user_col, item_col)


def run(
    get_recommendations: GetRecommendationsInput,
    items: pd.DataFrame,
    interactions: pd.DataFrame | None = None,
    layout: str = "rows",
    params: dict[str, Any] | None = None,
    config: str | None = None,
    title: str = "Recommender Demo",
    subtitle: str | None = None,
    item_columns: dict[str, str] | None = None,
    intro: Callable[[], None] | None = None,
    body: Callable[[], None] | None = None,
    session_user: bool = True,
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
    if intro:
        with st.container(border=True):
            intro()

    st.sidebar.subheader("User")
    user_id = _select_user(interactions, items, session_user=session_user)
    set_current_user(user_id)

    compare = is_compare_mode(get_recommendations)
    pairs = labeled_get_recommendations(get_recommendations)
    all_sections = [label for label, _ in pairs]

    st.sidebar.subheader("Parameters")
    global_params, model_param_defs = split_model_params(params, all_sections)
    resolved = resolve_params(global_params, yaml_params)
    model_params = resolve_model_params(
        all_sections,
        model_param_defs,
        cfg.get("model_params"),
    )
    k = int(resolved.pop("num_recs", resolved.pop("k", 20)))
    layout = resolved.pop("layout", layout)
    row_layout = "rows" if compare else layout
    sync_run_context(user_id, k, {"global": resolved, "models": model_params})

    selected_ids = get_selected_ids()
    selected_set = set(selected_ids)
    render_user_profile(interactions, user_id, items, session_items=selected_ids)

    history_ids = history_item_ids(interactions, user_id)
    if history_ids:
        render_profile_strip(items, history_ids, "Past interactions", columns=item_columns)
    render_profile_strip(
        items,
        selected_ids,
        "Selected this session",
        columns=item_columns,
        empty_hint="Click item cards below, then refresh recommendations.",
    )

    def _refresh_all_recommendations() -> None:
        fresh_selected_ids = get_selected_ids()
        for label, get_recommendations_fn in pairs:
            params_for_label = {**resolved, **model_params.get(label, {})}
            set_displayed_recs(
                label,
                cached_get_recommendations(
                    get_recommendations_fn,
                    user_id,
                    k,
                    params_for_label,
                    session_items=fresh_selected_ids,
                    selections=get_selections(label),
                ),
            )

    if compare:
        st.button(
            "Get Recommendations",
            type="primary",
            use_container_width=True,
            on_click=_refresh_all_recommendations,
        )

    for label, get_recommendations_fn in pairs:
        section_selections = get_selections(label)

        rec_ids = get_displayed_recs(label)
        if rec_ids is None:
            params_for_label = {**resolved, **model_params.get(label, {})}
            rec_ids = cached_get_recommendations(
                get_recommendations_fn,
                user_id,
                k,
                params_for_label,
                session_items=selected_ids,
                selections=section_selections,
            )
            set_displayed_recs(label, rec_ids)

        def _refresh_get_recommendations(
            label: str = label,
            get_recommendations_fn: Callable[..., list] = get_recommendations_fn,
        ) -> None:
            fresh_selected_ids = get_selected_ids()
            fresh_selections = get_selections(label)
            params_for_label = {**resolved, **model_params.get(label, {})}
            set_displayed_recs(
                label,
                cached_get_recommendations(
                    get_recommendations_fn,
                    user_id,
                    k,
                    params_for_label,
                    session_items=fresh_selected_ids,
                    selections=fresh_selections,
                ),
            )

        render_layout(
            row_layout,
            items,
            rec_ids,
            title=label,
            section=label,
            columns=item_columns,
            selected_ids=selected_set,
            all_sections=all_sections,
            on_get_recommendations=None if compare else _refresh_get_recommendations,
        )

    if body:
        with st.container(border=True):
            body()


def _select_user(
    interactions: pd.DataFrame | None,
    items: pd.DataFrame,
    *,
    session_user: bool = True,
) -> str | int:
    if interactions is not None and "user_id" in interactions.columns:
        users = sorted(interactions["user_id"].unique())
        options: list[str | int] = list(users)
        if session_user:
            options = [SESSION_USER_ID, *options]

        def _label(value: str | int) -> str:
            if value == SESSION_USER_ID:
                return SESSION_USER_LABEL
            return str(value)

        return st.sidebar.selectbox(
            "User",
            options,
            format_func=_label,
            key=user_select_key(),
            label_visibility="collapsed",
        )

    id_col = "item_id" if "item_id" in items.columns else items.columns[0]
    return st.sidebar.selectbox(
        "Context",
        items[id_col].head(20).tolist(),
        key=user_select_key(),
        label_visibility="collapsed",
    )
