from __future__ import annotations

import html

import streamlit as st

from streamlit_recommenders.layouts._helpers import item_placeholder
from streamlit_recommenders.runtime.keys import item_action_key
from streamlit_recommenders.runtime.state import record_selection

DEFAULT_GRID_COLS = 4
CARD_WIDTH_PX = 120
PROFILE_KEY_PREFIX = "streamlit_recommenders-profile-"


def _css_url(url: str) -> str:
    return html.escape(url, quote=True).replace("'", "%27")


def _css_string(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", " ")
        .replace("<", "")
    )


def _st_key_slug(key: str) -> str:
    return key.replace(".", "-")


def profile_poster_key(section: str, item_id: str | int, index: int) -> str:
    return f"{PROFILE_KEY_PREFIX}{section}-{index}-{item_id}"


def _strip_column_selector() -> str:
    item = '[class*="st-key-streamlit_recommenders-item-"]'
    profile = f'[class*="st-key-{PROFILE_KEY_PREFIX}"]'
    return (
        f'div[data-testid="stHorizontalBlock"]:has({item}), '
        f'div[data-testid="stHorizontalBlock"]:has({profile})'
    )


def _strip_column_rule() -> str:
    item = '[class*="st-key-streamlit_recommenders-item-"]'
    profile = f'[class*="st-key-{PROFILE_KEY_PREFIX}"]'
    width = f"{CARD_WIDTH_PX}px"
    return f"""
div[data-testid="stColumn"]:has({item}),
div[data-testid="stColumn"]:has({profile}) {{
    flex: 0 0 {width} !important;
    width: {width} !important;
    min-width: {width} !important;
    max-width: {width} !important;
}}
div[data-testid="stColumn"]:has({item}) div[data-testid="stVerticalBlock"],
div[data-testid="stColumn"]:has({profile}) div[data-testid="stVerticalBlock"] {{
    gap: 0 !important;
}}
"""


def _ensure_rec_card_styles() -> None:
    strip = _strip_column_selector()
    st.markdown(
        f"""
<style>
div[class*="st-key-streamlit_recommenders-recommend"] {{
    margin-top: 12px;
}}
{strip} {{
    display: flex !important;
    flex-wrap: nowrap !important;
    align-items: flex-start !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
    width: 100% !important;
    max-width: 100% !important;
    min-height: calc({CARD_WIDTH_PX}px * 1.5 + 14px) !important;
    padding: 0 2px 12px;
    scrollbar-width: thin;
}}
{_strip_column_rule()}
div[class*="st-key-{PROFILE_KEY_PREFIX}"] .sr-display-poster {{
    aspect-ratio: 2 / 3;
    width: 100%;
    border: 1px solid #d6dde8;
    border-radius: 8px;
    background-color: #eef2f7;
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    position: relative;
    overflow: hidden;
}}
div[class*="st-key-{PROFILE_KEY_PREFIX}"] .sr-display-poster::after {{
    content: attr(data-title);
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    padding: 10px 10px 12px;
    color: #fff;
    font-size: 0.72rem;
    font-weight: 700;
    line-height: 1.35;
    text-align: left;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    opacity: 0;
    transform: translateY(8px);
    transition: opacity 0.18s ease, transform 0.18s ease;
    background: linear-gradient(
        180deg,
        rgba(9, 15, 28, 0) 0%,
        rgba(9, 15, 28, 0.82) 45%,
        rgba(9, 15, 28, 0.96) 100%
    );
    pointer-events: none;
}}
div[class*="st-key-{PROFILE_KEY_PREFIX}"]:hover .sr-display-poster::after {{
    opacity: 1;
    transform: translateY(0);
}}
</style>
        """,
        unsafe_allow_html=True,
    )


def _button_card_css(entry: dict, card_id: str) -> str:
    kp = card_id
    image = _css_url(entry["image"] or item_placeholder())
    title = _css_string(entry["title"])
    selected = bool(entry.get("selected"))
    key_sel = f'div[class*="st-key-{kp}"]'
    selected_rules = ""
    if selected:
        selected_rules = f"""
{key_sel} button {{
    filter: grayscale(0.85);
    opacity: 0.62;
    border-color: #94a3b8 !important;
    cursor: default;
}}
{key_sel} button::after {{
    content: "Selected";
    position: absolute;
    top: 8px;
    right: 8px;
    padding: 4px 8px;
    border-radius: 999px;
    background: #475569;
    color: #fff;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    pointer-events: none;
}}
{key_sel} button:hover,
{key_sel} button:focus-visible {{
    transform: none;
    box-shadow: none;
}}
{key_sel} button::before {{
    opacity: 1;
    transform: translateY(0);
}}
"""
    return f"""
{key_sel} {{
    width: 100%;
}}
{key_sel} button {{
    aspect-ratio: 2 / 3;
    width: 100% !important;
    min-height: 0 !important;
    padding: 0 !important;
    border: 1px solid #d6dde8 !important;
    border-radius: 8px !important;
    background-color: #eef2f7 !important;
    background-image: url('{image}') !important;
    background-size: cover !important;
    background-position: center !important;
    background-repeat: no-repeat !important;
    color: transparent !important;
    position: relative;
    overflow: hidden;
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    cursor: pointer;
}}
{key_sel} button p {{
    display: none !important;
    margin: 0 !important;
    padding: 0 !important;
    font-size: 0 !important;
    line-height: 0 !important;
}}
{key_sel} button::before {{
    content: "{title}";
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    padding: 10px 10px 12px;
    color: #fff;
    font-size: clamp(0.62rem, 1.1vw, 0.78rem);
    font-weight: 700;
    line-height: 1.35;
    text-align: left;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    opacity: 0;
    transform: translateY(8px);
    transition: opacity 0.18s ease, transform 0.18s ease;
    background: linear-gradient(
        180deg,
        rgba(9, 15, 28, 0) 0%,
        rgba(9, 15, 28, 0.82) 45%,
        rgba(9, 15, 28, 0.96) 100%
    );
    pointer-events: none;
}}
{key_sel} button:hover,
{key_sel} button:focus-visible {{
    transform: translateY(-2px);
    border-color: #cbd5e1 !important;
    box-shadow: 0 10px 22px rgba(15, 23, 42, 0.08);
}}
{key_sel} button:hover::before,
{key_sel} button:focus-visible::before {{
    opacity: 1;
    transform: translateY(0);
}}
{key_sel} button:active {{
    transform: scale(0.985);
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.22);
}}
{key_sel} button:focus-visible {{
    outline: 2px solid #2563eb;
    outline-offset: 2px;
}}
{key_sel} button:disabled {{
    aspect-ratio: 2 / 3 !important;
    width: 100% !important;
    height: auto !important;
    min-height: calc({CARD_WIDTH_PX}px * 1.5) !important;
    padding: 0 !important;
}}
{selected_rules}
"""


def inject_card_styles(
    entries: list[dict],
    section: str,
    *,
    rank_offset: int = 0,
) -> None:
    """Inject CSS for poster-button cards (one rule per visible slot)."""
    _ensure_rec_card_styles()
    if not entries:
        return
    rules = [
        _button_card_css(
            entry,
            _st_key_slug(item_action_key(section, entry["id"], rank_offset + index)),
        )
        for index, entry in enumerate(entries)
    ]
    st.markdown(f"<style>{''.join(rules)}</style>", unsafe_allow_html=True)


def render_display_poster(entry: dict, key: str) -> None:
    """Non-interactive poster card for profile/history strips."""
    image = _css_url(entry["image"] or item_placeholder())
    title = html.escape(str(entry["title"]))
    with st.container(key=key):
        st.markdown(
            f'<div class="sr-display-poster" data-title="{title}" '
            f'style="background-image:url(\'{image}\');"></div>',
            unsafe_allow_html=True,
        )


def render_horizontal_posters(
    entries: list[dict],
    section: str,
    all_sections: list[str],
    *,
    rank_offset: int = 0,
    selectable: bool = True,
) -> None:
    """Fixed-width poster strip with horizontal scroll."""
    if not entries:
        return
    _ensure_rec_card_styles()
    if selectable:
        inject_card_styles(entries, section, rank_offset=rank_offset)
    cols = st.columns(len(entries), gap="small")
    for index, (col, entry) in enumerate(zip(cols, entries)):
        with col:
            if selectable:
                render_selectable_card(entry, rank_offset + index, section, all_sections)
            else:
                render_display_poster(entry, profile_poster_key(section, entry["id"], index))


def render_selectable_card(
    entry: dict,
    rank: int,
    section: str,
    all_sections: list[str],
) -> None:
    """Poster card rendered as a single styled button."""
    key = item_action_key(section, entry["id"], rank)
    description = str(entry.get("description") or entry["title"])
    if st.button(
        "\u200b",
        key=key,
        help=description,
        use_container_width=True,
        type="secondary",
        disabled=bool(entry.get("selected")),
    ):
        record_selection(section, entry["id"], rank, all_sections)
        st.rerun()
