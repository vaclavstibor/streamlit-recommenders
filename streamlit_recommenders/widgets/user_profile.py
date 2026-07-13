from __future__ import annotations

import pandas as pd
import streamlit as st

from streamlit_recommenders.runtime.seen import SESSION_USER_LABEL, is_session_user


def history_item_ids(
    interactions: pd.DataFrame | None,
    user_id: str | int,
) -> list:
    if interactions is None or "user_id" not in interactions.columns:
        return []
    if is_session_user(user_id):
        return []
    hist = interactions.loc[interactions["user_id"] == user_id, "item_id"]
    return hist.drop_duplicates().tolist()


def render_user_profile(
    interactions: pd.DataFrame | None,
    user_id: str | int,
    items: pd.DataFrame | None = None,
    session_items: list | None = None,
    *,
    max_items: int = 8,
) -> None:
    if interactions is None or "user_id" not in interactions.columns:
        if is_session_user(user_id):
            st.sidebar.caption("Session user — select items below to shape recommendations.")
        return

    session_items = session_items or []
    with st.sidebar.expander("Profile summary", expanded=False):
        if is_session_user(user_id):
            st.caption("Build your taste by selecting items in the recommendation rows.")
            st.markdown(f"Session picks: **{len(session_items)}**")
            return

        hist = interactions.loc[interactions["user_id"] == user_id]
        st.markdown(f"Past interactions: **{len(hist):,}**")
        st.markdown(f"Session additions: **{len(session_items)}**")
        if "rating" in hist.columns and len(hist):
            st.markdown(f"Average rating: **{hist['rating'].mean():.1f}**")
        if "timestamp" in hist.columns and len(hist) > 1:
            span_days = (hist["timestamp"].max() - hist["timestamp"].min()) / 86_400
            if span_days >= 1:
                st.markdown(f"Active span: **{span_days:,.0f} days**")
        if session_items:
            st.caption("New picks this session are merged into recommendations.")
