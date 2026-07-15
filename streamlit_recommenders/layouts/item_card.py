"""Poster-card rendering and injected CSS for item, profile, and swipe views.

Provides the low-level widgets and style injection used by the layouts:
selectable poster buttons, non-interactive display posters, and the
one-card swipe deck.
"""

from __future__ import annotations

import base64
import html
import mimetypes
from functools import lru_cache
from pathlib import Path

import streamlit as st

from streamlit_recommenders.layouts._helpers import item_placeholder
from streamlit_recommenders.runtime.keys import item_action_key, section_id
from streamlit_recommenders.runtime.state import record_selection

DEFAULT_GRID_COLS = 4
CARD_WIDTH_PX = 120
PROFILE_KEY_PREFIX = "streamlit_recommenders-profile-"
GRID_KEY_PREFIX = "streamlit_recommenders-grid-"
SWIPE_KEY_PREFIX = "streamlit_recommenders-swipe-"
SWIPE_POSTER_WIDTH_PX = 160
SWIPE_DECK_MAX_WIDTH_PX = 560


def _css_url(url: str) -> str:
    """Resolve an image url and escape it for safe use inside CSS ``url('')``."""
    return html.escape(_image_css_url(str(url)), quote=True).replace("'", "%27")


@lru_cache(maxsize=256)
def _image_css_url(url: str) -> str:
    """Return a usable image url, inlining local files as base64 data URIs.

    Remote (http/https) and existing data URIs pass through unchanged; local
    file paths are read and encoded; anything else falls back to the
    placeholder image.
    """
    if not url.strip():
        return _image_css_url(item_placeholder())
    if url.startswith(("http://", "https://", "data:")):
        return url

    path = Path(url)
    if path.is_file():
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    return _image_css_url(item_placeholder())


def _css_string(text: str) -> str:
    """Escape a string for safe embedding inside a double-quoted CSS value."""
    return (
        str(text)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", " ")
        .replace("<", "")
    )


def _st_key_slug(key: str) -> str:
    """Convert a state key into the CSS-class form Streamlit emits (dots to dashes)."""
    return key.replace(".", "-")


def profile_poster_key(section: str, item_id: str | int, index: int) -> str:
    """Return the container key for a profile/history poster slot."""
    return f"{PROFILE_KEY_PREFIX}{section}-{index}-{item_id}"


def _strip_column_selector() -> str:
    """Return the CSS selector matching the item/profile poster strip rows."""
    item = '[class*="st-key-streamlit_recommenders-item-"]'
    profile = f'[class*="st-key-{PROFILE_KEY_PREFIX}"]'
    return (
        f'div[data-testid="stHorizontalBlock"]:has({item}), '
        f'div[data-testid="stHorizontalBlock"]:has({profile})'
    )


def _strip_column_rule() -> str:
    """Return the CSS pinning each poster column to a fixed card width."""
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
    """Inject the base strip/column and profile-poster CSS for a section."""
    strip = _strip_column_selector()
    _inject_html(
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
    min-height: calc({CARD_WIDTH_PX}px * 1.5 + 26px) !important;
    padding: 12px 2px 12px;
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
    )


def _button_card_css(entry: dict, card_id: str) -> str:
    """Return the CSS for one poster-button card, incl. its selected state."""
    kp = card_id
    image = _css_url(entry["image"] or item_placeholder())
    selected = bool(entry.get("selected"))
    key_sel = f'div[class*="st-key-{kp}"]'
    selected_rules = ""
    if selected:
        selected_rules = f"""
{key_sel} button {{
    opacity: 1 !important;
    border-color: #2563eb !important;
    box-shadow: inset 0 0 0 2px rgba(37, 99, 235, 0.85);
    cursor: pointer;
}}
{key_sel} button::before {{
    opacity: 1;
    background: linear-gradient(
        180deg,
        rgba(9, 15, 28, 0.18) 0%,
        rgba(9, 15, 28, 0.34) 38%,
        rgba(9, 15, 28, 0.94) 100%
    );
}}
{key_sel} button::after {{
    content: "Selected";
    position: absolute;
    top: 8px;
    right: 8px;
    padding: 4px 8px;
    border-radius: 999px;
    background: #2563eb;
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
    box-shadow: inset 0 0 0 2px rgba(37, 99, 235, 0.85);
}}
{key_sel} .sr-card-copy {{
    opacity: 1;
    transform: translateY(0);
}}
"""
    return f"""
{key_sel} {{
    width: 100%;
    position: relative;
}}
{key_sel} button {{
    aspect-ratio: 2 / 3;
    width: 100% !important;
    min-height: calc({CARD_WIDTH_PX}px * 1.5) !important;
    height: auto !important;
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
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    top: 0;
    bottom: 0;
    opacity: 0;
    transition: opacity 0.18s ease;
    background: linear-gradient(
        180deg,
        rgba(9, 15, 28, 0) 0%,
        rgba(9, 15, 28, 0.58) 48%,
        rgba(9, 15, 28, 0.96) 100%
    );
    pointer-events: none;
}}
{key_sel} .sr-card-copy {{
    position: absolute;
    left: 10px;
    right: 10px;
    bottom: 10px;
    color: #fff;
    text-align: left;
    opacity: 0;
    transform: translateY(8px);
    transition: opacity 0.18s ease, transform 0.18s ease;
    pointer-events: none;
    z-index: 2;
}}
{key_sel} .sr-card-title {{
    display: block;
    font-size: clamp(0.66rem, 1.1vw, 0.8rem);
    font-weight: 800;
    line-height: 1.2;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
{key_sel} .sr-card-desc {{
    display: block;
    margin-top: 3px;
    font-size: clamp(0.56rem, 0.95vw, 0.68rem);
    font-weight: 500;
    line-height: 1.25;
    opacity: 0.9;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
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
}}
{key_sel}:hover .sr-card-copy,
{key_sel}:focus-within .sr-card-copy {{
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
    """Inject CSS for poster-button cards (one rule per visible slot).

    Args:
        entries: Card entries whose slots need styling.
        section: Stable section identifier for state keys.
        rank_offset: Offset added to each slot index so keys stay unique when
            entries span multiple rows/strips.
    """
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
    _inject_html(f"<style>{''.join(rules)}</style>")


def _inject_html(html: str) -> None:
    """Emit raw HTML/CSS via ``st.html`` when available, else ``st.markdown``."""
    if hasattr(st, "html"):
        st.html(html)
    else:
        st.markdown(html, unsafe_allow_html=True)


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


def swipe_deck_key(section: str) -> str:
    """Return the container key for a section's swipe deck."""
    return f"{SWIPE_KEY_PREFIX}deck-{section}"


def render_swipe_card(entry: dict) -> None:
    """Poster on the left, title and description on the right."""
    image = _css_url(entry["image"] or item_placeholder())
    title = html.escape(str(entry["title"]))
    description = html.escape(str(entry.get("description") or ""))
    description_html = (
        f'<div class="sr-swipe-description">{description}</div>' if description else ""
    )
    st.markdown(
        (
            '<div class="sr-swipe-card">'
            f'<div class="sr-swipe-poster" style="background-image:url(\'{image}\');"></div>'
            '<div class="sr-swipe-details">'
            f'<div class="sr-swipe-title">{title}</div>'
            f"{description_html}"
            "</div></div>"
        ),
        unsafe_allow_html=True,
    )


def inject_swipe_styles(
    deck_key: str,
    dislike_key: str,
    skip_key: str,
    like_key: str,
) -> None:
    """Centered deck layout plus Dislike (red) / Skip (blue) / Like (green) button colors."""
    deck = _st_key_slug(deck_key)
    dislike = _st_key_slug(dislike_key)
    skip = _st_key_slug(skip_key)
    like = _st_key_slug(like_key)
    _inject_html(
        f"""
<style>
div[class*="st-key-{deck}"] {{
    max-width: {SWIPE_DECK_MAX_WIDTH_PX}px;
    margin: 0 auto;
}}
div[class*="st-key-{deck}"] .sr-swipe-card {{
    display: flex;
    align-items: center;
    gap: 20px;
    padding: 4px 0 12px;
}}
div[class*="st-key-{deck}"] .sr-swipe-poster {{
    flex: 0 0 {SWIPE_POSTER_WIDTH_PX}px;
    width: {SWIPE_POSTER_WIDTH_PX}px;
    aspect-ratio: 2 / 3;
    border-radius: 12px;
    background-color: #eef2f7;
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.1);
}}
div[class*="st-key-{deck}"] .sr-swipe-details {{
    flex: 1;
    min-width: 0;
    text-align: left;
}}
div[class*="st-key-{deck}"] .sr-swipe-title {{
    font-weight: 700;
    font-size: 1.05rem;
    line-height: 1.35;
    color: #0f172a;
}}
div[class*="st-key-{deck}"] .sr-swipe-description {{
    margin-top: 8px;
    color: #64748b;
    font-size: 0.875rem;
    line-height: 1.45;
}}
{_swipe_button_rule(dislike, "#dc2626", "#b91c1c")}
{_swipe_button_rule(skip, "#2563eb", "#1d4ed8")}
{_swipe_button_rule(like, "#16a34a", "#15803d")}
</style>
        """,
    )


def _swipe_button_rule(key_slug: str, color: str, hover: str) -> str:
    """Return CSS coloring one swipe button in its base and hover states."""
    sel = f'div[class*="st-key-{key_slug}"] button'
    return f"""
{sel} {{
    background-color: {color} !important;
    border-color: {color} !important;
    color: #ffffff !important;
}}
{sel}:hover,
{sel}:focus-visible {{
    background-color: {hover} !important;
    border-color: {hover} !important;
    color: #ffffff !important;
}}
{sel} p {{ color: #ffffff !important; font-weight: 700 !important; }}
"""


def render_horizontal_posters(
    entries: list[dict],
    section: str,
    all_sections: list[str],
    *,
    rank_offset: int = 0,
    selectable: bool = True,
) -> None:
    """Render a fixed-width poster strip with horizontal scroll.

    Args:
        entries: Card entries to render, one column each.
        section: Stable section identifier for state keys.
        all_sections: Every section id on the page.
        rank_offset: Offset added to each card's rank for unique keys.
        selectable: When ``True`` draw clickable selection cards; otherwise
            draw non-interactive display posters.
    """
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


def grid_container_key(section: str) -> str:
    """Return the container key that wraps a grid so it scrolls as one unit."""
    return f"{GRID_KEY_PREFIX}{section_id(section)}"


def _inject_grid_scroll_styles(grid_key: str) -> None:
    """Scroll the whole grid horizontally as one unit instead of per row.

    Overrides the shared strip rule (which scrolls each row on its own and
    clamps it to the container width) so the wrapper owns the single scrollbar
    and the rows size to their content, moving together.
    """
    wrapper = f'div[class*="st-key-{_st_key_slug(grid_key)}"]'
    rows = f'{wrapper} div[data-testid="stHorizontalBlock"]'
    _inject_html(
        f"""
<style>
{wrapper} {{
    overflow-x: auto !important;
    overflow-y: hidden !important;
    scrollbar-width: thin;
}}
{rows} {{
    overflow-x: visible !important;
    overflow-y: visible !important;
    width: max-content !important;
    min-width: 100% !important;
    max-width: none !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
}}
</style>
        """,
    )


def render_grid_posters(
    entries: list[dict],
    section: str,
    all_sections: list[str],
    *,
    n_cols: int = DEFAULT_GRID_COLS,
    selectable: bool = True,
) -> None:
    """Render a wrapped poster gallery for browsing-style layouts.

    Args:
        entries: Card entries to lay out row by row.
        section: Stable section identifier for state keys.
        all_sections: Every section id on the page.
        n_cols: Posters per row; clamped to at least 1.
        selectable: When ``True`` draw clickable selection cards; otherwise
            draw non-interactive display posters.
    """
    if not entries:
        return
    n_cols = max(1, n_cols)
    _ensure_rec_card_styles()
    if selectable:
        inject_card_styles(entries, section)

    grid_key = grid_container_key(section)
    _inject_grid_scroll_styles(grid_key)
    with st.container(key=grid_key):
        for start in range(0, len(entries), n_cols):
            row_entries = entries[start : start + n_cols]
            cols = st.columns(n_cols, gap="small")
            for index, col in enumerate(cols):
                if index >= len(row_entries):
                    continue
                entry = row_entries[index]
                with col:
                    rank = start + index
                    if selectable:
                        render_selectable_card(entry, rank, section, all_sections)
                    else:
                        render_display_poster(entry, profile_poster_key(section, entry["id"], rank))


def render_selectable_card(
    entry: dict,
    rank: int,
    section: str,
    all_sections: list[str],
) -> None:
    """Render a poster card as a single styled button that records selection.

    Args:
        entry: The card entry to render.
        rank: The card's position, used for its state key and the recorded
            selection.
        section: Stable section identifier for state keys.
        all_sections: Every section id on the page, passed to
            ``record_selection``.
    """
    key = item_action_key(section, entry["id"], rank)
    title = str(entry["title"])
    description = str(entry.get("description") or entry["title"])
    tooltip = f"{title}\n\n{description}" if description != title else title
    with st.container(key=key):
        st.button(
            "\u200b",
            key=f"{key}.button",
            help=tooltip,
            use_container_width=True,
            type="secondary",
            on_click=record_selection,
            args=(section, entry["id"], rank, all_sections),
        )
        st.markdown(
            (
                '<div class="sr-card-copy">'
                f'<span class="sr-card-title">{html.escape(str(entry["title"]))}</span>'
                f'<span class="sr-card-desc">{html.escape(description)}</span>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )
