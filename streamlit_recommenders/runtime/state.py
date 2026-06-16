import streamlit as st

STATE_KEY = "_sr_state"


def init_session_state() -> None:
    if STATE_KEY not in st.session_state:
        st.session_state[STATE_KEY] = {
            "current_user": None,
            "clicked_items": [],
            "params_snapshot": {},
        }


def get_state() -> dict:
    init_session_state()
    return st.session_state[STATE_KEY]


def set_current_user(user_id: str | int) -> None:
    state = get_state()
    if state["current_user"] != user_id:
        state["current_user"] = user_id
        state["clicked_items"] = []


def record_click(item_id: str | int) -> None:
    state = get_state()
    if item_id not in state["clicked_items"]:
        state["clicked_items"].append(item_id)


def get_clicked_items() -> list:
    return list(get_state()["clicked_items"])
