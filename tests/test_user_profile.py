import pandas as pd

from streamlit_recommenders.runtime.seen import SESSION_USER_ID
from streamlit_recommenders.widgets.user_profile import history_item_ids


def test_history_item_ids():
    interactions = pd.DataFrame({"user_id": [0, 0, 1], "item_id": [3, 7, 1]})
    assert history_item_ids(interactions, 0) == [3, 7]


def test_history_item_ids_session_user():
    interactions = pd.DataFrame({"user_id": [0], "item_id": [1]})
    assert history_item_ids(interactions, SESSION_USER_ID) == []
