"""Public API for streamlit_recommenders."""

from streamlit_recommenders.runner import load_interactions, load_items, run
from streamlit_recommenders.content.markdown import markdown, markdown_file
from streamlit_recommenders.data import ColumnMap, Dataset, load_dataset, load_users, validate_dataset
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
    EASERecommender,
    ELSARecommender,
    EmbeddingPopularityRecommender,
    ItemKNNRecommender,
    PopularityRecommender,
    RandomRecommender,
    SequentialCFRecommender,
)
from streamlit_recommenders.runtime.keys import param_key
from streamlit_recommenders.runtime.state import get_selected_ids, get_selections, get_state
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
    return get_state()["current_user"]


def selected_items():
    return get_selected_ids()


def selections(section: str | None = None):
    return get_selections(section)


def param_value(name: str, default=None):
    import streamlit as st

    return st.session_state.get(param_key(name), default)


__all__ = [
    "run",
    "load_items",
    "load_interactions",
    "load_users",
    "load_dataset",
    "validate_dataset",
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
    "EASERecommender",
    "ELSARecommender",
    "EmbeddingPopularityRecommender",
    "ItemKNNRecommender",
    "PopularityRecommender",
    "RandomRecommender",
    "SequentialCFRecommender",
    "current_user",
    "selected_items",
    "selections",
    "param_value",
    "hit_rate_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "mrr_at_k",
    "coverage",
    "evaluate",
]
