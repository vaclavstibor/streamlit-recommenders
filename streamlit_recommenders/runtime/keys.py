"""Streamlit widget keys — single namespace for the library."""

from __future__ import annotations

import re

NS = "streamlit_recommenders"


def section_id(label: str) -> str:
    """Slugify a section label into a key-safe identifier.

    Args:
        label: Human-readable section label.

    Returns:
        A lowercase underscore slug, or ``"default"`` when empty.
    """
    slug = re.sub(r"[^\w]+", "_", label.strip().lower()).strip("_")
    return slug or "default"


def item_action_key(section: str, item_id: str | int, rank: int | None = None) -> str:
    """Build the widget key for an item's action control.

    Args:
        section: Section the item is rendered in.
        item_id: Identifier of the item.
        rank: Optional rank, appended to disambiguate repeated items.

    Returns:
        The namespaced widget key.
    """
    base = f"{NS}.item.{section_id(section)}.{item_id}"
    if rank is not None:
        return f"{base}.r{rank}"
    return base


def param_key(name: str) -> str:
    """Return the namespaced widget key for a model parameter control.

    Args:
        name: Parameter name.

    Returns:
        The namespaced widget key.
    """
    return f"{NS}.param.{name}"


def user_select_key() -> str:
    """Return the widget key for the user selector.

    Returns:
        The namespaced widget key.
    """
    return f"{NS}.user"


def get_recommendations_button_key(section: str) -> str:
    """Return the widget key for a section's "get recommendations" button.

    Args:
        section: Section the button belongs to.

    Returns:
        The namespaced widget key.
    """
    return f"{NS}.get_recommendations.{section_id(section)}"
