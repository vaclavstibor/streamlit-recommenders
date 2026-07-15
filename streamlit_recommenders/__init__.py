"""Public API for streamlit_recommenders."""

from streamlit_recommenders.runner import (
    load_artifacts,
    load_interactions,
    load_items,
    load_local_dataset,
    run,
)
from streamlit_recommenders.content.markdown import markdown, markdown_file
from streamlit_recommenders.data import (
    ColumnMap,
    Dataset,
    load_dataset,
    load_users,
    resolve_image_urls,
    validate_dataset,
)
from streamlit_recommenders.data.prepare import (
    is_complete,
    prepare_goodbooks,
    prepare_movielens,
)
from streamlit_recommenders.layouts import render_layout
from streamlit_recommenders.layouts.cards import cards
from streamlit_recommenders.layouts.grid import grid
from streamlit_recommenders.layouts.rows import rows
from streamlit_recommenders.metrics import (
    coverage,
    evaluate,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from streamlit_recommenders.models.protocol import RecommenderProtocol
from streamlit_recommenders.recommenders import (
    ArtifactRecommender,
    BaseRecommender,
)
from streamlit_recommenders.runtime.keys import param_key
from streamlit_recommenders.runtime.state import (
    get_disliked_ids,
    get_selected_ids,
    get_selections,
    get_state,
)
from streamlit_recommenders.viz.plot import (
    plot,
    plot_overlap_heatmap,
    plot_metric_comparison,
    plot_ranked_items,
    plot_score_distribution,
    recommendation_overlap_matrix,
)
from streamlit_recommenders.viz.dataset_info import dataset_info
from streamlit_recommenders.viz.table import table
from streamlit_recommenders.widgets.params import selectbox, slider


def current_user():
    """Return the id of the currently selected user."""
    return get_state()["current_user"]


def selected_items():
    """Return the item ids the user selected during this session."""
    return get_selected_ids()


def disliked_items():
    """Return the item ids the user disliked during this session."""
    return get_disliked_ids()


def displayed_items(section: str | None = None):
    """Item ids currently shown in the recommender window.

    Returns the list for one ``section`` label, or a ``{label: ids}`` dict for
    all sections when ``section`` is ``None``. This is the single source of
    truth for what the UI displays, so diagnostics in ``body()`` (overlap
    heatmaps, score tables) stay consistent with the rows on screen, including
    the cold-start sample shown for the "Try yourself" session user.
    """
    displayed = get_state().get("displayed_recs", {})
    if section is None:
        return {label: list(ids) for label, ids in displayed.items()}
    ids = displayed.get(section)
    return list(ids) if ids is not None else []


def selections(section: str | None = None):
    """Return recorded like/dislike selections for a section.

    Args:
        section: Section label to fetch selections for, or ``None`` for all
            sections.

    Returns:
        The selection metadata for the requested section(s).
    """
    return get_selections(section)


def param_value(name: str, default=None):
    """Return the current sidebar value for parameter ``name``.

    Args:
        name: Parameter name as defined in the sidebar controls.
        default: Value to return if the parameter is not set.

    Returns:
        The parameter's current value from session state, or ``default``.
    """
    import streamlit as st

    return st.session_state.get(param_key(name), default)


__all__ = [
    "run",
    "load_items",
    "load_interactions",
    "load_users",
    "load_dataset",
    "load_local_dataset",
    "resolve_image_urls",
    "validate_dataset",
    "prepare_movielens",
    "prepare_goodbooks",
    "is_complete",
    "ColumnMap",
    "Dataset",
    "slider",
    "selectbox",
    "rows",
    "grid",
    "cards",
    "render_layout",
    "plot",
    "plot_overlap_heatmap",
    "plot_metric_comparison",
    "plot_ranked_items",
    "plot_score_distribution",
    "recommendation_overlap_matrix",
    "dataset_info",
    "table",
    "markdown",
    "markdown_file",
    "RecommenderProtocol",
    "ArtifactRecommender",
    "BaseRecommender",
    "load_artifacts",
    "current_user",
    "selected_items",
    "disliked_items",
    "displayed_items",
    "selections",
    "param_value",
    "hit_rate_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "mrr_at_k",
    "coverage",
    "evaluate",
]
