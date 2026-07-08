import streamlit as st

from streamlit_recommenders.runtime.cache import hash_params

STATE_KEY = "_sr_state"


def init_session_state() -> None:
    if STATE_KEY not in st.session_state:
        st.session_state[STATE_KEY] = {
            "current_user": None,
            "selected_ids": [],
            "selections": {},
            "displayed_recs": {},
            "run_context_hash": None,
        }


def get_state() -> dict:
    init_session_state()
    return st.session_state[STATE_KEY]


def set_current_user(user_id: str | int) -> None:
    state = get_state()
    if state["current_user"] != user_id:
        state["current_user"] = user_id
        _reset_session(state)


def sync_run_context(user_id: str | int, k: int, params: dict) -> None:
    state = get_state()
    ctx_hash = hash_params({"user_id": user_id, "k": k, **params})
    if state.get("run_context_hash") != ctx_hash:
        state["run_context_hash"] = ctx_hash
        state["displayed_recs"] = {}


def _reset_session(state: dict) -> None:
    state["selected_ids"] = []
    state["selections"] = {}
    state["displayed_recs"] = {}


def get_selected_ids() -> list:
    return list(get_state()["selected_ids"])


def get_displayed_recs(section: str) -> list | None:
    displayed = get_state().get("displayed_recs", {})
    recs = displayed.get(section)
    return list(recs) if recs is not None else None


def set_displayed_recs(section: str, rec_ids: list) -> None:
    get_state().setdefault("displayed_recs", {})[section] = list(rec_ids)


def get_selections(section: str | None = None) -> list[dict] | dict[str, list[dict]]:
    selections: dict[str, list[dict]] = get_state()["selections"]
    if section is None:
        return {key: list(items) for key, items in selections.items()}
    return list(selections.get(section, []))


def record_selection(
    source_section: str,
    item_id: str | int,
    rank: int,
    all_sections: list[str],
) -> None:
    state = get_state()
    if item_id in state["selected_ids"]:
        state["selected_ids"] = [selected_id for selected_id in state["selected_ids"] if selected_id != item_id]
        for section in all_sections:
            selections = state["selections"].get(section, [])
            state["selections"][section] = [
                selection for selection in selections if selection.get("item_id") != item_id
            ]
        return
    state["selected_ids"].append(item_id)
    for section in all_sections:
        state["selections"].setdefault(section, []).append(
            {
                "item_id": item_id,
                "rank": rank if section == source_section else None,
                "source": source_section,
            }
        )
