import streamlit as st

from streamlit_recommenders.runtime.cache import hash_params

STATE_KEY = "_sr_state"


def init_session_state() -> None:
    if STATE_KEY not in st.session_state:
        st.session_state[STATE_KEY] = {
            "current_user": None,
            "selected_ids": [],
            "disliked_ids": [],
            "selections": {},
            "displayed_recs": {},
            "swipe_counts": {},
            "swipe_skipped": {},
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
        state["swipe_counts"] = {}
        state["swipe_skipped"] = {}


def _reset_session(state: dict) -> None:
    state["selected_ids"] = []
    state["disliked_ids"] = []
    state["selections"] = {}
    state["displayed_recs"] = {}
    state["swipe_counts"] = {}
    state["swipe_skipped"] = {}


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


def get_disliked_ids() -> list:
    return list(get_state().get("disliked_ids", []))


def record_swipe(
    source_section: str,
    item_id: str | int,
    sentiment: str,
    all_sections: list[str],
) -> None:
    """Record a like or dislike swipe. Likes mirror a card click (positive profile)."""
    if sentiment == "like":
        record_selection(source_section, item_id, 0, all_sections)
        return

    state = get_state()
    disliked = state.setdefault("disliked_ids", [])
    if item_id in disliked:
        return
    disliked.append(item_id)
    for section in all_sections:
        state["selections"].setdefault(section, []).append(
            {
                "item_id": item_id,
                "rank": None,
                "source": source_section,
                "sentiment": "dislike",
            }
        )


def record_skip(section: str, item_id: str | int) -> None:
    state = get_state()
    skipped = state.setdefault("swipe_skipped", {}).setdefault(section, [])
    if item_id not in skipped:
        skipped.append(item_id)


def get_swipe_skipped(section: str) -> list:
    return list(get_state().get("swipe_skipped", {}).get(section, []))


def get_swipe_seen_ids(section: str) -> list:
    """All items already shown in a swipe deck during the current session."""
    state = get_state()
    return list(
        dict.fromkeys(
            [
                *state.get("selected_ids", []),
                *state.get("disliked_ids", []),
                *state.get("swipe_skipped", {}).get(section, []),
            ]
        )
    )


def bump_swipe_count(section: str) -> int:
    counts = get_state().setdefault("swipe_counts", {})
    counts[section] = counts.get(section, 0) + 1
    return counts[section]


def reset_swipe_count(section: str) -> None:
    state = get_state()
    state.setdefault("swipe_counts", {})[section] = 0
