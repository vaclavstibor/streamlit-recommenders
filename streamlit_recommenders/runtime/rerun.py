from streamlit_recommenders.runtime.cache import hash_params


def params_changed(old: dict, new: dict) -> bool:
    return hash_params(old) != hash_params(new)
