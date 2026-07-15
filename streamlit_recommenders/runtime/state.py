"""Session-state accessors and mutators for the recommender UI.

All per-user interaction state (selections, dislikes, swipe progress, the
recommendations actually displayed, and the cold-start seed) lives under a
single ``STATE_KEY`` slot in ``st.session_state`` and is reached exclusively
through the helpers in this module.
"""

import random

import streamlit as st

from streamlit_recommenders.runtime.cache import hash_params

STATE_KEY = "_sr_state"


def init_session_state() -> None:
    """Create the state slot with default values if it does not yet exist."""
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
            "cold_start_seed": None,
        }


def get_cold_start_seed() -> int:
    """Return the per-session seed for cold-start sampling.

    Lazily generated once per session and cached in state, so cold-start
    results stay stable across reruns within a session but differ between
    sessions (a fresh page load starts a new session and a new seed).

    Returns:
        The 32-bit random seed for this session.
    """
    state = get_state()
    if state.get("cold_start_seed") is None:
        state["cold_start_seed"] = random.randrange(2**32)
    return state["cold_start_seed"]


def get_state() -> dict:
    """Return the session-state dict, initializing it on first access.

    Returns:
        The mutable per-session state dict stored under ``STATE_KEY``.
    """
    init_session_state()
    return st.session_state[STATE_KEY]


def set_current_user(user_id: str | int) -> None:
    """Switch the active user, resetting session interactions when it changes.

    Args:
        user_id: Identifier of the user to make active.
    """
    state = get_state()
    if state["current_user"] != user_id:
        state["current_user"] = user_id
        _reset_session(state)


def sync_run_context(user_id: str | int, k: int, params: dict) -> None:
    """Clear displayed recs and swipe progress when the run context changes.

    The run context is a hash of the user, ``k``, and model params. When it
    differs from the last run, cached display/swipe state no longer matches the
    request and is reset so the next render reflects the new context.

    Args:
        user_id: Identifier of the active user.
        k: Number of recommendations requested.
        params: Model parameters that influence the recommendation request.
    """
    state = get_state()
    ctx_hash = hash_params({"user_id": user_id, "k": k, **params})
    if state.get("run_context_hash") != ctx_hash:
        state["run_context_hash"] = ctx_hash
        state["displayed_recs"] = {}
        state["swipe_counts"] = {}
        state["swipe_skipped"] = {}


def _reset_session(state: dict) -> None:
    """Clear all interaction state in place (keeps the current user)."""
    state["selected_ids"] = []
    state["disliked_ids"] = []
    state["selections"] = {}
    state["displayed_recs"] = {}
    state["swipe_counts"] = {}
    state["swipe_skipped"] = {}


def get_selected_ids() -> list:
    """Return a copy of the liked/selected item ids for the session.

    Returns:
        The selected item ids in insertion order.
    """
    return list(get_state()["selected_ids"])


def get_displayed_recs(section: str) -> list | None:
    """Return the recommendation ids last shown in a section, if any.

    ``displayed_recs`` records the exact ids rendered per section so actions
    and re-renders operate on the same frozen list rather than re-querying.

    Args:
        section: Section label whose displayed recommendations to fetch.

    Returns:
        A copy of the displayed ids, or ``None`` if the section has none yet.
    """
    displayed = get_state().get("displayed_recs", {})
    recs = displayed.get(section)
    return list(recs) if recs is not None else None


def set_displayed_recs(section: str, rec_ids: list) -> None:
    """Store the recommendation ids currently shown in a section.

    Args:
        section: Section label the recommendations belong to.
        rec_ids: Item ids rendered in that section.
    """
    get_state().setdefault("displayed_recs", {})[section] = list(rec_ids)


def get_selections(section: str | None = None) -> list[dict] | dict[str, list[dict]]:
    """Return recorded selection entries, for one section or all of them.

    Args:
        section: Section label to fetch; if ``None``, return every section.

    Returns:
        A copy of the selection entries for ``section``, or a copy of the full
        ``{section: entries}`` mapping when ``section`` is ``None``.
    """
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
    """Toggle an item's selection, updating ids and per-section entries.

    If the item is already selected it is removed everywhere; otherwise it is
    appended to ``selected_ids`` and a selection entry is added under every
    section, carrying its rank only for the section it was clicked in.

    Args:
        source_section: Section the item was selected from.
        item_id: Identifier of the item.
        rank: Rank of the item within ``source_section``.
        all_sections: All section labels to mirror the selection across.
    """
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
    """Return a copy of the disliked item ids for the session.

    Returns:
        The disliked item ids in insertion order.
    """
    return list(get_state().get("disliked_ids", []))


def record_swipe(
    source_section: str,
    item_id: str | int,
    sentiment: str,
    all_sections: list[str],
) -> None:
    """Record a like or dislike swipe.

    Likes mirror a card click (routed through :func:`record_selection` to feed
    the positive profile). Dislikes are appended to ``disliked_ids`` and logged
    as ``sentiment="dislike"`` selection entries across all sections.

    Args:
        source_section: Section the swipe originated from.
        item_id: Identifier of the swiped item.
        sentiment: Either ``"like"`` or ``"dislike"``.
        all_sections: All section labels to mirror the swipe across.
    """
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
    """Mark an item as skipped (neither liked nor disliked) in a section.

    Args:
        section: Section the skip happened in.
        item_id: Identifier of the skipped item.
    """
    state = get_state()
    skipped = state.setdefault("swipe_skipped", {}).setdefault(section, [])
    if item_id not in skipped:
        skipped.append(item_id)


def get_swipe_skipped(section: str) -> list:
    """Return the items skipped in a swipe section.

    Args:
        section: Section label to query.

    Returns:
        A copy of the skipped item ids for that section.
    """
    return list(get_state().get("swipe_skipped", {}).get(section, []))


def get_swipe_seen_ids(section: str) -> list:
    """Return every item already shown in a swipe deck this session.

    Unions selected, disliked, and skipped ids (de-duplicated, order-preserving)
    so the deck never re-surfaces a card the user has already acted on.

    Args:
        section: Section whose skipped items are included.

    Returns:
        The deduplicated seen item ids for that section.
    """
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
    """Increment and return the swipe counter for a section.

    Args:
        section: Section whose counter to increment.

    Returns:
        The counter value after incrementing.
    """
    counts = get_state().setdefault("swipe_counts", {})
    counts[section] = counts.get(section, 0) + 1
    return counts[section]


def reset_swipe_count(section: str) -> None:
    """Reset a section's swipe counter to zero.

    Args:
        section: Section whose counter to reset.
    """
    state = get_state()
    state.setdefault("swipe_counts", {})[section] = 0
