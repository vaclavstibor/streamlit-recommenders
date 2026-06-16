import hashlib
import json
from collections.abc import Callable
from typing import Any

import pandas as pd
import streamlit as st

from streamlit_recommenders.models.adapter import call_recommend


def hash_params(params: dict[str, Any]) -> str:
    """Stable hash for recommend() cache keys."""
    payload = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(payload.encode()).hexdigest()


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def cached_recommend(
    recommend_id: int,
    user_id: str | int,
    k: int,
    params_hash: str,
    _recommend: Callable[..., list],
    **params: Any,
) -> tuple:
    """Cache recommend() output; params_hash drives invalidation on slider change."""
    return tuple(call_recommend(_recommend, user_id, k, **params))


def get_recommendations(
    recommend: Callable[..., list],
    user_id: str | int,
    k: int,
    params: dict[str, Any],
) -> list:
    params_hash = hash_params(params)
    recommend_id = id(recommend)
    result = cached_recommend(
        recommend_id,
        user_id,
        k,
        params_hash,
        recommend,
        **params,
    )
    return list(result)
