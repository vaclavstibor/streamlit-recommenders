from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import pandas as pd

SESSION_USER_ID = "__session__"
SESSION_USER_LABEL = "Try yourself (session)"


def is_session_user(user_id: str | int) -> bool:
    return user_id == SESSION_USER_ID


def effective_seen(
    interactions: pd.DataFrame | None,
    user_id: str | int,
    session_items: list | None = None,
) -> set[str | int]:
    seen: set[str | int] = set(session_items or [])
    if (
        interactions is not None
        and "user_id" in interactions.columns
        and not is_session_user(user_id)
    ):
        seen.update(interactions.loc[interactions.user_id == user_id, "item_id"])
    return seen


def selection_item_ids(selections: list[dict] | None) -> list:
    if not selections:
        return []
    return [entry["item_id"] for entry in selections]


def call_with_session_context(
    get_recommendations: Callable[..., list],
    user_id: str | int,
    k: int,
    session_items: list,
    selections: list[dict] | None,
    **params: Any,
) -> list:
    payload = dict(params)
    if session_items:
        payload["session_items"] = list(session_items)
    if selections is not None:
        payload["selections"] = list(selections)
    filtered = filter_model_params(get_recommendations, payload)
    return get_recommendations(user_id, k, **filtered)


def filter_model_params(fn: Callable[..., list], params: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return params
    allowed = {
        name
        for name, p in sig.parameters.items()
        if name not in {"user_id", "k", "self"} and p.kind != inspect.Parameter.VAR_POSITIONAL
    }
    return {key: value for key, value in params.items() if key in allowed}
