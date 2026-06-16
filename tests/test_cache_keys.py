from streamlit_recommenders.runtime.cache import hash_params


def test_hash_params_stable_types():
    h1 = hash_params({"k": 10, "alpha": 0.5})
    h2 = hash_params({"k": 10, "alpha": 0.5})
    assert h1 == h2


def test_hash_params_different_k():
    assert hash_params({"k": 5}) != hash_params({"k": 10})
