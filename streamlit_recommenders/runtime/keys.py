"""Streamlit widget keys — single namespace for the library."""

from __future__ import annotations

import re

NS = "streamlit_recommenders"


def section_id(label: str) -> str:
    slug = re.sub(r"[^\w]+", "_", label.strip().lower()).strip("_")
    return slug or "default"


def item_action_key(section: str, item_id: str | int, rank: int | None = None) -> str:
    base = f"{NS}.item.{section_id(section)}.{item_id}"
    if rank is not None:
        return f"{base}.r{rank}"
    return base


def param_key(name: str) -> str:
    return f"{NS}.param.{name}"


def user_select_key() -> str:
    return f"{NS}.user"


def get_recommendations_button_key(section: str) -> str:
    return f"{NS}.get_recommendations.{section_id(section)}"
