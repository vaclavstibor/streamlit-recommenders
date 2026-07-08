from streamlit_recommenders.runtime.state import (
    STATE_KEY,
    get_selected_ids,
    get_selections,
    init_session_state,
    record_selection,
)


class _FakeSessionState(dict):
    pass


def test_record_selection_propagates_to_all_sections(monkeypatch):
    fake = _FakeSessionState()
    fake[STATE_KEY] = {
        "current_user": 0,
        "selected_ids": [],
        "selections": {},
        "params_snapshot": {},
    }
    monkeypatch.setattr(
        "streamlit_recommenders.runtime.state.st.session_state",
        fake,
    )

    record_selection("Model A", 5, 1, ["Model A", "Model B"])
    assert get_selected_ids() == [5]
    assert get_selections("Model A") == [{"item_id": 5, "rank": 1, "source": "Model A"}]
    assert get_selections("Model B") == [{"item_id": 5, "rank": None, "source": "Model A"}]


def test_record_selection_toggles_existing_item(monkeypatch):
    fake = _FakeSessionState()
    fake[STATE_KEY] = {
        "current_user": 0,
        "selected_ids": [5],
        "selections": {
            "Model A": [{"item_id": 5, "rank": 0, "source": "Model A"}],
            "Model B": [{"item_id": 5, "rank": None, "source": "Model A"}],
        },
        "params_snapshot": {},
    }
    monkeypatch.setattr(
        "streamlit_recommenders.runtime.state.st.session_state",
        fake,
    )

    record_selection("Model B", 5, 2, ["Model A", "Model B"])
    assert get_selected_ids() == []
    assert get_selections("Model A") == []
    assert get_selections("Model B") == []


def test_record_selection_unselect_preserves_other_items(monkeypatch):
    fake = _FakeSessionState()
    fake[STATE_KEY] = {
        "current_user": 0,
        "selected_ids": [5, 7],
        "selections": {
            "Model A": [
                {"item_id": 5, "rank": 0, "source": "Model A"},
                {"item_id": 7, "rank": 1, "source": "Model A"},
            ],
            "Model B": [
                {"item_id": 5, "rank": None, "source": "Model A"},
                {"item_id": 7, "rank": None, "source": "Model A"},
            ],
        },
        "params_snapshot": {},
    }
    monkeypatch.setattr(
        "streamlit_recommenders.runtime.state.st.session_state",
        fake,
    )

    record_selection("Model A", 5, 0, ["Model A", "Model B"])

    assert get_selected_ids() == [7]
    assert get_selections("Model A") == [{"item_id": 7, "rank": 1, "source": "Model A"}]
    assert get_selections("Model B") == [{"item_id": 7, "rank": None, "source": "Model A"}]
