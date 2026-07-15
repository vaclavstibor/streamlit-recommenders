"""Seen-item computation and recommender call plumbing.

Determines which items a user has already encountered (to exclude from
recommendations) and adapts session context onto arbitrary recommender
callables by filtering out kwargs the callable does not accept.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import pandas as pd

SESSION_USER_ID = "__session__"
SESSION_USER_LABEL = "Try yourself (session)"


def is_session_user(user_id: str | int) -> bool:
    """Return whether ``user_id`` is the synthetic session ("try yourself") user.

    Args:
        user_id: Identifier to test.

    Returns:
        ``True`` if it is the session user sentinel, else ``False``.
    """
    return user_id == SESSION_USER_ID


def effective_seen(
    interactions: pd.DataFrame | None,
    user_id: str | int,
    session_items: list | None = None,
) -> set[str | int]:
    """Return the set of items a user has already seen.

    Unions the in-session items with the user's historical interactions. For
    the session user there is no history, so only ``session_items`` contribute.

    Args:
        interactions: Interaction log with ``user_id``/``item_id`` columns, or
            ``None`` when no history is available.
        user_id: Identifier of the user to compute seen items for.
        session_items: Items already engaged with during this session.

    Returns:
        The union of session and historical item ids.
    """
    seen: set[str | int] = set(session_items or [])
    if (
        interactions is not None
        and "user_id" in interactions.columns
        and not is_session_user(user_id)
    ):
        seen.update(interactions.loc[interactions.user_id == user_id, "item_id"])
    return seen


def selection_item_ids(selections: list[dict] | None) -> list:
    """Extract item ids from a list of selection entries.

    Args:
        selections: Selection entries, each with an ``item_id`` key, or ``None``.

    Returns:
        The item ids in order, or an empty list when there are none.
    """
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
    """Call a recommender, injecting session context it is able to accept.

    ``session_items`` and ``selections`` are added to the params only when
    present, then :func:`filter_model_params` drops any that the callable's
    signature does not declare, so simpler recommenders are called cleanly.

    Args:
        get_recommendations: The recommender callable to invoke.
        user_id: Identifier of the active user.
        k: Number of recommendations to request.
        session_items: Items engaged with during the session.
        selections: Recorded selection entries, or ``None``.
        **params: Additional model parameters.

    Returns:
        The recommender's returned list of item ids.
    """
    payload = dict(params)
    if session_items:
        payload["session_items"] = list(session_items)
    if selections is not None:
        payload["selections"] = list(selections)
    filtered = filter_model_params(get_recommendations, payload)
    return get_recommendations(user_id, k, **filtered)


def filter_model_params(fn: Callable[..., list], params: dict[str, Any]) -> dict[str, Any]:
    """Keep only the params a callable can accept as keyword arguments.

    If the callable takes ``**kwargs`` every param is passed through. Otherwise
    only names declared in its signature are kept, excluding ``user_id``, ``k``,
    ``self``, and any ``*args`` variadic parameter.

    Args:
        fn: The callable whose signature constrains the params.
        params: Candidate keyword arguments.

    Returns:
        The subset of ``params`` accepted by ``fn``.
    """
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return params
    allowed = {
        name
        for name, p in sig.parameters.items()
        if name not in {"user_id", "k", "self"} and p.kind != inspect.Parameter.VAR_POSITIONAL
    }
    return {key: value for key, value in params.items() if key in allowed}
