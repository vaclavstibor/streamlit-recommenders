import pandas as pd

from streamlit_recommenders.viz.plot import (
    plot,
    plot_metric_comparison,
    plot_overlap_heatmap,
    plot_ranked_items,
    plot_score_distribution,
    recommendation_overlap_matrix,
)


def test_plot_functions_render_without_error():
    df = pd.DataFrame({"x": ["a", "b"], "y": [1.0, 2.0]})
    plot(df, x="x", y="y")
    plot_metric_comparison(
        pd.DataFrame({"metric": ["ndcg", "ndcg"], "value": [0.1, 0.2], "model": ["A", "B"]})
    )
    plot_ranked_items(pd.DataFrame({"title": ["a", "b"], "score": [2.0, 1.0]}))
    plot_score_distribution(pd.DataFrame({"score": [0.1, 0.5, 0.9]}))


def test_overlap_heatmap_renders_with_layout_overrides():
    # plot_overlap_heatmap overrides the default margin; guards against
    # duplicate-kwarg regressions in the shared _style helper.
    overlap = recommendation_overlap_matrix({"A": [1, 2, 3], "B": [2, 3, 4]})
    plot_overlap_heatmap(overlap)


def test_empty_frames_short_circuit():
    empty = pd.DataFrame()
    plot(empty)
    plot_metric_comparison(empty)
    plot_ranked_items(empty)
    plot_score_distribution(empty)
    plot_overlap_heatmap(empty)
