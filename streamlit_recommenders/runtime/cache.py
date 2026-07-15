"""Streamlit-backed caching for data loads and recommendation calls.

Wraps recommender invocations in ``st.cache_data`` keyed by a stable model
identity plus a hash of the request context, so repeated reruns reuse results
instead of recomputing them.
"""

import hashlib
import json
from collections.abc import Callable
from typing import Any

import pandas as pd
import streamlit as st

from streamlit_recommenders.models.adapter import call_get_recommendations


def hash_params(params: dict[str, Any]) -> str:
    """Return a stable MD5 hash of a params dict for cache keying.

    Args:
        params: JSON-serializable params (non-serializable values are stringified).

    Returns:
        The hex digest of the canonically sorted JSON payload.
    """
    payload = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(payload.encode(), usedforsecurity=False).hexdigest()


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV into a DataFrame, cached by path.

    Args:
        path: Filesystem path to the CSV file.

    Returns:
        The parsed DataFrame.
    """
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
    """Return cached recommendations for a fully-hashed request context.

    The leading string/tuple/hash args form the cache key; ``_get_recommendations``
    is prefixed with an underscore so Streamlit excludes the (unhashable)
    callable from the key, relying on ``get_recommendations_id`` for identity.

    Args:
        get_recommendations_id: Stable identity of the recommender callable.
        user_id: Identifier of the active user.
        k: Number of recommendations to request.
        params_hash: Hash of the full request context (part of the cache key).
        session_items: Items engaged with during the session.
        selections_json: JSON-encoded selection entries, or empty string.
        _get_recommendations: The recommender callable (excluded from the key).
        **params: Additional model parameters.

    Returns:
        The recommended item ids as a tuple (hashable for caching).
    """
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
    excluded_item_ids: list | None = None,
) -> list:
    """Fetch recommendations through the cache, applying exclusions post-hoc.

    Requests ``k + len(excluded)`` items from the cached call so that removing
    excluded ids still leaves up to ``k`` results. Exclusion is applied after
    the cache lookup rather than pushed into the model call.

    Args:
        get_recommendations_fn: The recommender callable to invoke.
        user_id: Identifier of the active user.
        k: Number of recommendations to return.
        params: Model parameters influencing the request.
        session_items: Items engaged with during the session.
        selections: Recorded selection entries, or ``None``.
        excluded_item_ids: Item ids to drop from the results (e.g. already seen).

    Returns:
        Up to ``k`` recommended item ids with exclusions removed.
    """
    session_tuple = tuple(session_items or [])
    excluded_tuple = tuple(excluded_item_ids or [])
    selections_json = json.dumps(selections or [], sort_keys=True, default=str)
    params_hash = hash_params(
        {
            **params,
            "session_items": session_tuple,
            "selections": selections_json,
            "excluded_item_ids": excluded_tuple,
        }
    )
    request_k = k + len(excluded_tuple)
    result = cached_get_recommendations(
        stable_get_recommendations_id(get_recommendations_fn),
        user_id,
        request_k,
        params_hash,
        session_tuple,
        selections_json,
        get_recommendations_fn,
        **params,
    )
    excluded = set(excluded_tuple)
    return [item_id for item_id in result if item_id not in excluded][:k]


def stable_get_recommendations_id(get_recommendations_fn: Callable[..., list]) -> str:
    """Identity used in the Streamlit cache key for a recommender callable.

    Bound methods stay stable across reruns as long as the owning model
    instance is reused (e.g. constructed under ``st.cache_resource``). Plain
    callables fall back to object identity, so closures recreated on every
    rerun miss the cache by design (safe, but slower) — prefer module-level
    functions or cached model objects.

    Args:
        get_recommendations_fn: The recommender callable to identify.

    Returns:
        A string combining object identity and qualified name for cache keying.
    """
    owner = getattr(get_recommendations_fn, "__self__", None)
    func = getattr(get_recommendations_fn, "__func__", None)
    if owner is not None and func is not None:
        return f"{id(owner)}:{func.__module__}.{func.__qualname__}"
    qualname = getattr(
        get_recommendations_fn, "__qualname__", type(get_recommendations_fn).__name__
    )
    module = getattr(get_recommendations_fn, "__module__", "")
    return f"{id(get_recommendations_fn)}:{module}.{qualname}"
