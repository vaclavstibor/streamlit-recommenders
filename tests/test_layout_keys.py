from streamlit_recommenders.runtime.keys import (
    item_action_key,
    param_key,
    recommend_button_key,
    section_id,
    user_select_key,
)


def test_section_id():
    assert section_id("Our method") == "our_method"
    assert section_id("Popularity baseline") == "popularity_baseline"
    assert section_id("---") == "default"


def test_item_action_key_unique_per_section():
    assert item_action_key("Our method", 23) == "streamlit_recommenders.item.our_method.23"
    assert item_action_key("Popularity baseline", 23) != item_action_key("Our method", 23)
    assert item_action_key("Our method", 23, 0) == "streamlit_recommenders.item.our_method.23.r0"
    assert item_action_key("Our method", 23, 1) != item_action_key("Our method", 23, 0)


def test_param_key():
    assert param_key("alpha") == "streamlit_recommenders.param.alpha"


def test_recommend_button_key():
    assert recommend_button_key("Our method") == "streamlit_recommenders.recommend.our_method"
