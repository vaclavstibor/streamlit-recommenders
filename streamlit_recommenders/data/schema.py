from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from streamlit_recommenders.runtime.cache import load_csv


@dataclass(frozen=True)
class ColumnMap:
    """Logical column names used by the library."""

    item_id: str = "item_id"
    user_id: str = "user_id"
    rating: str = "rating"
    timestamp: str = "timestamp"
    title: str = "title"
    image: str = "image_url"
    description: str = "description"

    def item_columns(self) -> dict[str, str]:
        return {
            "id": self.item_id,
            "title": self.title,
            "image": self.image,
            "description": self.description,
        }


@dataclass
class Dataset:
    """Small bundle for common recommender inputs."""

    items: pd.DataFrame
    interactions: pd.DataFrame | None = None
    users: pd.DataFrame | None = None
    train: pd.DataFrame | None = None
    test: pd.DataFrame | None = None
    columns: ColumnMap = field(default_factory=ColumnMap)

    def validate(self) -> Dataset:
        validate_dataset(self)
        return self


def load_items(path: str, id_col: str = "item_id") -> pd.DataFrame:
    df = load_csv(path)
    require_columns(df, [id_col], "items")
    return df


def load_interactions(path: str) -> pd.DataFrame:
    df = load_csv(path)
    require_columns(df, ["user_id", "item_id"], "interactions")
    return df


def load_users(path: str, id_col: str = "user_id") -> pd.DataFrame:
    df = load_csv(path)
    require_columns(df, [id_col], "users")
    return df


def load_dataset(
    *,
    items: str,
    interactions: str | None = None,
    users: str | None = None,
    train: str | None = None,
    test: str | None = None,
    columns: ColumnMap | None = None,
) -> Dataset:
    cols = columns or ColumnMap()
    dataset = Dataset(
        items=load_items(items, cols.item_id),
        interactions=load_interactions(interactions) if interactions else None,
        users=load_users(users, cols.user_id) if users else None,
        train=load_interactions(train) if train else None,
        test=load_interactions(test) if test else None,
        columns=cols,
    )
    return dataset.validate()


def validate_dataset(dataset: Dataset) -> None:
    cols = dataset.columns
    require_columns(dataset.items, [cols.item_id], "items")

    item_ids = set(dataset.items[cols.item_id])
    _validate_table_ids(dataset.interactions, "interactions", cols, item_ids)
    _validate_table_ids(dataset.train, "train", cols, item_ids)
    _validate_table_ids(dataset.test, "test", cols, item_ids)

    if dataset.users is not None:
        require_columns(dataset.users, [cols.user_id], "users")
        user_ids = set(dataset.users[cols.user_id])
        for name, table in (
            ("interactions", dataset.interactions),
            ("train", dataset.train),
            ("test", dataset.test),
        ):
            if table is not None:
                unknown = set(table[cols.user_id]) - user_ids
                if unknown:
                    raise ValueError(f"{name} contains unknown user ids: {sorted(unknown)[:5]}")


def require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _validate_table_ids(
    table: pd.DataFrame | None,
    name: str,
    cols: ColumnMap,
    item_ids: set,
) -> None:
    if table is None:
        return
    require_columns(table, [cols.user_id, cols.item_id], name)
    unknown = set(table[cols.item_id]) - item_ids
    if unknown:
        raise ValueError(f"{name} contains unknown item ids: {sorted(unknown)[:5]}")
