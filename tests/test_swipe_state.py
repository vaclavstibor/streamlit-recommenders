from streamlit_recommenders.runtime.state import (
    STATE_KEY,
    bump_swipe_count,
    get_disliked_ids,
    get_selected_ids,
    get_selections,
    get_swipe_seen_ids,
    get_swipe_skipped,
    record_skip,
    record_swipe,
    reset_swipe_count,
)


class _FakeSessionState(dict):
    pass


def _fresh_state(monkeypatch) -> None:
    fake = _FakeSessionState()
    fake[STATE_KEY] = {
        "current_user": 0,
        "selected_ids": [],
        "disliked_ids": [],
        "selections": {},
        "displayed_recs": {},
        "swipe_counts": {},
        "swipe_skipped": {},
        "run_context_hash": None,
    }
    monkeypatch.setattr("streamlit_recommenders.runtime.state.st.session_state", fake)


def test_like_swipe_adds_to_session_profile(monkeypatch):
    _fresh_state(monkeypatch)

    record_swipe("Deck", 5, "like", ["Deck"])

    assert get_selected_ids() == [5]
    assert get_disliked_ids() == []
    assert get_selections("Deck") == [{"item_id": 5, "rank": 0, "source": "Deck"}]


def test_dislike_swipe_records_negative_signal(monkeypatch):
    _fresh_state(monkeypatch)

    record_swipe("Deck", 7, "dislike", ["Deck"])

    assert get_selected_ids() == []
    assert get_disliked_ids() == [7]
    assert get_selections("Deck") == [
        {"item_id": 7, "rank": None, "source": "Deck", "sentiment": "dislike"}
    ]


def test_skip_is_tracked_per_section(monkeypatch):
    _fresh_state(monkeypatch)

    record_skip("Deck", 9)

    assert get_swipe_skipped("Deck") == [9]
    assert get_selected_ids() == []
    assert get_disliked_ids() == []
    assert get_swipe_seen_ids("Deck") == [9]


def test_swipe_count_bumps_and_resets(monkeypatch):
    _fresh_state(monkeypatch)

    assert bump_swipe_count("Deck") == 1
    assert bump_swipe_count("Deck") == 2
    record_skip("Deck", 3)

    reset_swipe_count("Deck")

    assert bump_swipe_count("Deck") == 1
    assert get_swipe_skipped("Deck") == [3]


def test_seen_swipe_ids_include_all_actions(monkeypatch):
    _fresh_state(monkeypatch)

    record_swipe("Deck", 1, "like", ["Deck"])
    record_swipe("Deck", 2, "dislike", ["Deck"])
    record_skip("Deck", 3)

    assert get_swipe_seen_ids("Deck") == [1, 2, 3]
