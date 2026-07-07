from streamlit_recommenders.runtime.keys import (
    get_recommendations_button_key,
    item_action_key,
    param_key,
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


def test_get_recommendations_button_key():
    assert (
        get_recommendations_button_key("Our method")
        == "streamlit_recommenders.get_recommendations.our_method"
    )
