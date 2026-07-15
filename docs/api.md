# API reference

Generated from the source docstrings. Start with `sr.run`; the recommender contract is
`get_recommendations(user_id, k, session_items=None, selections=None, **params) -> list[item_id]`.

## Application

::: streamlit_recommenders.run

## Data loading

::: streamlit_recommenders.load_items
::: streamlit_recommenders.load_interactions
::: streamlit_recommenders.load_dataset
::: streamlit_recommenders.load_dataset_from_paths

## Session readers (for `body()`)

::: streamlit_recommenders.current_user
::: streamlit_recommenders.selected_items
::: streamlit_recommenders.disliked_items
::: streamlit_recommenders.displayed_items
::: streamlit_recommenders.selections
::: streamlit_recommenders.param_value

## Recommender contract

::: streamlit_recommenders.recommenders.base.BaseRecommender

::: streamlit_recommenders.recommenders.artifact.ArtifactRecommender

::: streamlit_recommenders.load_artifacts

::: streamlit_recommenders.models.protocol.RecommenderProtocol

## Metrics

::: streamlit_recommenders.evaluate

## Visualization

::: streamlit_recommenders.dataset_info
