import random
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
    get_cold_start_seed,
    get_displayed_recs,
    get_selected_ids,
    get_selections,
    get_swipe_seen_ids,
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

# Cold-start sampling: how many times k to request before sampling k.
_COLD_START_POOL_FACTOR = 5


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
    swipes_per_refresh: int = 5,
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
    swipes_per_refresh = int(resolved.pop("swipes_per_refresh", swipes_per_refresh))
    row_layout = "rows" if compare else layout
    if row_layout == "cards":
        k = max(k, swipes_per_refresh + 1)
    sync_run_context(user_id, k, {"global": resolved, "models": model_params})

    if row_layout == "cards":
        st.sidebar.caption(
            f"Swipe to like or dislike. New recommendations after {swipes_per_refresh} swipes."
        )

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
        selectable=True,
        all_sections=all_sections,
    )

    def _compute_recommendations(label: str, get_recommendations_fn: Callable[..., list]) -> list:
        current_selected = get_selected_ids()
        section_selections = get_selections(label)
        params_for_label = {**resolved, **model_params.get(label, {})}
        # The session user's empty profile would otherwise always show the
        # same deterministic fallback; sample from a larger candidate pool
        # so each fresh session starts with varied items to react to.
        cold_start = (
            user_id == SESSION_USER_ID
            and not current_selected
            and not section_selections
        )
        request_k = k * _COLD_START_POOL_FACTOR if cold_start else k
        rec_ids = cached_get_recommendations(
            get_recommendations_fn,
            user_id,
            request_k,
            params_for_label,
            session_items=current_selected,
            selections=section_selections,
            excluded_item_ids=get_swipe_seen_ids(label) if row_layout == "cards" else None,
        )
        if cold_start and len(rec_ids) > k:
            # Seed per section: each compared model starts with a different
            # sample, so the init state is not the same list repeated n times.
            rng = random.Random(f"{get_cold_start_seed()}:{label}")
            rec_ids = rng.sample(list(rec_ids), k)
        return rec_ids

    def _refresh_all_recommendations() -> None:
        for label, get_recommendations_fn in pairs:
            set_displayed_recs(label, _compute_recommendations(label, get_recommendations_fn))

    if compare:
        st.button(
            "Get Recommendations",
            type="primary",
            use_container_width=True,
            on_click=_refresh_all_recommendations,
        )

    for label, get_recommendations_fn in pairs:
        rec_ids = get_displayed_recs(label)
        if rec_ids is None:
            rec_ids = _compute_recommendations(label, get_recommendations_fn)
            set_displayed_recs(label, rec_ids)

        def _refresh_get_recommendations(
            label: str = label,
            get_recommendations_fn: Callable[..., list] = get_recommendations_fn,
        ) -> None:
            set_displayed_recs(label, _compute_recommendations(label, get_recommendations_fn))

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
            swipes_per_refresh=swipes_per_refresh,
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
