import pandas as pd
import pytest

from streamlit_recommenders.data import ColumnMap, Dataset, validate_dataset


def test_dataset_validation_accepts_items_interactions_users():
    dataset = Dataset(
        items=pd.DataFrame({"item_id": [1, 2], "title": ["A", "B"]}),
        interactions=pd.DataFrame({"user_id": [10], "item_id": [1]}),
        users=pd.DataFrame({"user_id": [10]}),
    )

    validate_dataset(dataset)


def test_dataset_validation_rejects_unknown_item_id():
    dataset = Dataset(
        items=pd.DataFrame({"item_id": [1]}),
        interactions=pd.DataFrame({"user_id": [10], "item_id": [2]}),
    )

    with pytest.raises(ValueError, match="unknown item ids"):
        validate_dataset(dataset)


def test_column_map_exports_layout_mapping():
    columns = ColumnMap(item_id="movie_id", image="poster")

    assert columns.item_columns() == {
        "id": "movie_id",
        "title": "title",
        "image": "poster",
        "description": "description",
    }


def test_dataset_validation_uses_column_map_for_interactions():
    columns = ColumnMap(item_id="movie_id", user_id="viewer_id")
    dataset = Dataset(
        items=pd.DataFrame({"movie_id": [1, 2], "title": ["A", "B"]}),
        interactions=pd.DataFrame({"viewer_id": [10], "movie_id": [1]}),
        users=pd.DataFrame({"viewer_id": [10]}),
        columns=columns,
    )

    validate_dataset(dataset)
