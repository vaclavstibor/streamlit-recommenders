import hashlib
import json
from collections.abc import Callable
from typing import Any

import pandas as pd
import streamlit as st

from streamlit_recommenders.models.adapter import call_get_recommendations


def hash_params(params: dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(payload.encode()).hexdigest()


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def cached_get_recommendations(
    get_recommendations_id: str,
    user_id: str | int,
    k: int,
    params_hash: str,
    session_items: tuple,
    selections_json: str,
    _get_recommendations: Callable[..., list],
    **params: Any,
) -> tuple:
    selections = json.loads(selections_json) if selections_json else []
    return tuple(
        call_get_recommendations(
            _get_recommendations,
            user_id,
            k,
            list(session_items),
            selections,
            **params,
        )
    )


def get_recommendations(
    get_recommendations_fn: Callable[..., list],
    user_id: str | int,
    k: int,
    params: dict[str, Any],
    session_items: list | None = None,
    selections: list[dict] | None = None,
) -> list:
    session_tuple = tuple(session_items or [])
    selections_json = json.dumps(selections or [], sort_keys=True, default=str)
    params_hash = hash_params(
        {**params, "session_items": session_tuple, "selections": selections_json}
    )
    result = cached_get_recommendations(
        stable_get_recommendations_id(get_recommendations_fn),
        user_id,
        k,
        params_hash,
        session_tuple,
        selections_json,
        get_recommendations_fn,
        **params,
    )
    return list(result)


def stable_get_recommendations_id(get_recommendations_fn: Callable[..., list]) -> str:
    owner = getattr(get_recommendations_fn, "__self__", None)
    func = getattr(get_recommendations_fn, "__func__", None)
    if owner is not None and func is not None:
        return f"{id(owner)}:{func.__module__}.{func.__qualname__}"
    return str(id(get_recommendations_fn))
