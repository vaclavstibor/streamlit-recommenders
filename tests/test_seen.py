import pandas as pd

from streamlit_recommenders.runtime.seen import SESSION_USER_ID, effective_seen, is_session_user
from streamlit_recommenders.widgets.user_profile import history_item_ids


def test_effective_seen_merges_session_items():
    interactions = pd.DataFrame({"user_id": [0, 0], "item_id": [1, 2]})
    seen = effective_seen(interactions, 0, session_items=[3])
    assert seen == {1, 2, 3}


def test_effective_seen_session_user_skips_history():
    interactions = pd.DataFrame({"user_id": [0], "item_id": [1]})
    seen = effective_seen(interactions, SESSION_USER_ID, session_items=[4])
    assert seen == {4}


def test_history_item_ids_session_user_empty():
    interactions = pd.DataFrame({"user_id": [0], "item_id": [1]})
    assert history_item_ids(interactions, SESSION_USER_ID) == []


def test_is_session_user():
    assert is_session_user(SESSION_USER_ID)
    assert not is_session_user(0)
