"""CSV schema, dataset container, and loaders for recommender inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from streamlit_recommenders.runtime.cache import load_csv


@dataclass(frozen=True)
class ColumnMap:
    """Logical column names used by the library.

    Only item_id is required for item metadata. Display fields such as title,
    image, and description are optional and fall back in the UI.

    Attributes:
        item_id: Column holding the item identifier.
        user_id: Column holding the user identifier.
        rating: Column holding the interaction rating.
        timestamp: Column holding the interaction timestamp.
        title: Column holding the item display title.
        image: Column holding the item poster/image reference.
        description: Column holding the item description text.
    """

    item_id: str = "item_id"
    user_id: str = "user_id"
    rating: str = "rating"
    timestamp: str = "timestamp"
    title: str = "title"
    image: str = "image_url"
    description: str = "description"

    def item_columns(self) -> dict[str, str]:
        """Map display roles to their configured item column names."""
        return {
            "id": self.item_id,
            "title": self.title,
            "image": self.image,
            "description": self.description,
        }


@dataclass
class Dataset:
    """Small bundle for common recommender inputs.

    Attributes:
        items: Item metadata table (required).
        interactions: Full user-item interactions, if provided.
        users: User metadata table, if provided.
        train: Training split of interactions, if provided.
        test: Test split of interactions, if provided.
        columns: Logical-to-physical column name mapping.
    """

    items: pd.DataFrame
    interactions: pd.DataFrame | None = None
    users: pd.DataFrame | None = None
    train: pd.DataFrame | None = None
    test: pd.DataFrame | None = None
    columns: ColumnMap = field(default_factory=ColumnMap)

    def validate(self) -> Dataset:
        """Validate the dataset in place and return self for chaining.

        Raises:
            ValueError: If required columns are missing or ids are unknown.
        """
        validate_dataset(self)
        return self


def load_items(path: str, id_col: str = "item_id") -> pd.DataFrame:
    """Load an items CSV and require the id column.

    Args:
        path: Path to the items CSV.
        id_col: Name of the required item id column.

    Returns:
        The loaded items DataFrame.

    Raises:
        ValueError: If the id column is missing.
    """
    df = load_csv(path)
    require_columns(df, [id_col], "items")
    return df


def load_interactions(
    path: str,
    user_col: str = "user_id",
    item_col: str = "item_id",
) -> pd.DataFrame:
    """Load an interactions CSV and require the user and item columns.

    Args:
        path: Path to the interactions CSV.
        user_col: Name of the required user id column.
        item_col: Name of the required item id column.

    Returns:
        The loaded interactions DataFrame.

    Raises:
        ValueError: If a required column is missing.
    """
    df = load_csv(path)
    require_columns(df, [user_col, item_col], "interactions")
    return df


def load_users(path: str, id_col: str = "user_id") -> pd.DataFrame:
    """Load a users CSV and require the id column.

    Args:
        path: Path to the users CSV.
        id_col: Name of the required user id column.

    Returns:
        The loaded users DataFrame.

    Raises:
        ValueError: If the id column is missing.
    """
    df = load_csv(path)
    require_columns(df, [id_col], "users")
    return df


def load_dataset_from_paths(
    *,
    items: str,
    interactions: str | None = None,
    users: str | None = None,
    train: str | None = None,
    test: str | None = None,
    columns: ColumnMap | None = None,
) -> Dataset:
    """Load and validate a Dataset from individual CSV paths.

    Args:
        items: Path to the items CSV.
        interactions: Optional path to the full interactions CSV.
        users: Optional path to the users CSV.
        train: Optional path to the training interactions CSV.
        test: Optional path to the test interactions CSV.
        columns: Column mapping to apply; defaults to ``ColumnMap()``.

    Returns:
        A validated Dataset.

    Raises:
        ValueError: If validation fails.
    """
    cols = columns or ColumnMap()
    dataset = Dataset(
        items=load_items(items, cols.item_id),
        interactions=load_interactions(interactions, cols.user_id, cols.item_id)
        if interactions
        else None,
        users=load_users(users, cols.user_id) if users else None,
        train=load_interactions(train, cols.user_id, cols.item_id) if train else None,
        test=load_interactions(test, cols.user_id, cols.item_id) if test else None,
        columns=cols,
    )
    return dataset.validate()


def load_dataset(
    root: str | Path,
    *,
    resolve_images: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Load items + interactions (+ test) from a prepared local dataset folder.

    Expects ``items.csv`` and either ``train_interactions.csv`` or
    ``interactions.csv``; ``test_interactions.csv`` is optional. When
    ``resolve_images`` is set, item poster paths are resolved to local files.

    Args:
        root: Prepared dataset folder.
        resolve_images: Resolve item poster paths to local files when True.

    Returns:
        Tuple of ``(items, interactions, test)``; the latter two are None
        when their CSVs are absent.
    """
    root = Path(root)
    items = load_items(str(root / "items.csv"))
    if resolve_images:
        from streamlit_recommenders.data.media import resolve_image_urls

        items = items.copy()
        items["image_url"] = resolve_image_urls(items, root)

    train_path = root / "train_interactions.csv"
    interactions_path = root / "interactions.csv"
    if train_path.exists():
        interactions = load_interactions(str(train_path))
    elif interactions_path.exists():
        interactions = load_interactions(str(interactions_path))
    else:
        interactions = None

    test_path = root / "test_interactions.csv"
    test = load_interactions(str(test_path)) if test_path.exists() else None
    return items, interactions, test


def validate_dataset(dataset: Dataset) -> None:
    """Check required columns and cross-table id consistency.

    Args:
        dataset: The dataset to validate.

    Raises:
        ValueError: If required columns are missing, or interaction tables
            reference unknown item or user ids.
    """
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
    """Raise if any required column is absent from ``df``.

    Args:
        df: DataFrame to check.
        columns: Required column names.
        name: Table label used in the error message.

    Raises:
        ValueError: If one or more columns are missing.
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _validate_table_ids(
    table: pd.DataFrame | None,
    name: str,
    cols: ColumnMap,
    item_ids: set,
) -> None:
    """Require id columns and check the table references only known items."""
    if table is None:
        return
    require_columns(table, [cols.user_id, cols.item_id], name)
    unknown = set(table[cols.item_id]) - item_ids
    if unknown:
        raise ValueError(f"{name} contains unknown item ids: {sorted(unknown)[:5]}")
