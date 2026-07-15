"""Prepare the goodbooks-10k dataset (books carry cover image URLs directly)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from streamlit_recommenders.data.prepare.manifest import is_complete, write_manifest
from streamlit_recommenders.data.prepare.movielens import require_columns

KAGGLE_DATASET = "zygmunt/goodbooks-10k"


def prepare_goodbooks(
    root: str | Path | None = None,
    *,
    force: bool = False,
) -> Path:
    """Build items.csv/interactions.csv from goodbooks-10k.

    Uses local ``books.csv``/``ratings.csv`` when present, otherwise downloads via
    ``kagglehub`` if it is installed. Cover ``image_url`` ships with the dataset.

    Args:
        root: Output folder; defaults to ``data/goodbooks-10k``.
        force: Rebuild even if the folder is already marked complete.

    Returns:
        Path to the prepared dataset folder.

    Raises:
        SystemExit: If the CSVs are absent and ``kagglehub`` is not installed.
    """
    root = Path(root) if root else Path("data") / "goodbooks-10k"
    root.mkdir(parents=True, exist_ok=True)

    if is_complete(root) and not force:
        print(f"{root} already prepared (complete). Use force=True to rebuild.")
        return root

    books_csv = root / "books.csv"
    ratings_csv = root / "ratings.csv"
    if not books_csv.exists() or not ratings_csv.exists():
        _download_goodbooks(root)

    items, interactions = _build_goodbooks(books_csv, ratings_csv)
    items.to_csv(root / "items.csv", index=False)
    interactions.to_csv(root / "interactions.csv", index=False)
    write_manifest(
        root,
        name="goodbooks-10k",
        n_items=int(len(items)),
        n_interactions=int(len(interactions)),
        with_posters=True,
    )
    print(f"Prepared {len(items):,} items and {len(interactions):,} interactions in {root}")
    return root


def _build_goodbooks(books_csv: Path, ratings_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build items/interactions tables from goodbooks CSVs, keying on ``item_id``."""
    books = pd.read_csv(books_csv)
    ratings = pd.read_csv(ratings_csv)
    require_columns(books, ["book_id", "title"], "books.csv")
    require_columns(ratings, ["user_id", "book_id", "rating"], "ratings.csv")

    rename = {"book_id": "item_id", "image_url": "image_url", "authors": "authors"}
    keep = [col for col in ["book_id", "title", "authors", "image_url"] if col in books.columns]
    items = books[keep].rename(columns=rename)

    interactions = ratings.rename(columns={"book_id": "item_id"})[["user_id", "item_id", "rating"]]
    return items, interactions


def _download_goodbooks(root: Path) -> None:
    """Download goodbooks CSVs into ``root`` via kagglehub, or exit if unavailable."""
    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "goodbooks-10k not found locally. Either place books.csv/ratings.csv in "
            f'{root}, or install kagglehub (pip install "streamlit-recommenders[goodbooks]") '
            f"to auto-download {KAGGLE_DATASET}."
        ) from exc

    print(f"Downloading {KAGGLE_DATASET} via kagglehub")
    source = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    for name in ("books.csv", "ratings.csv"):
        matches = list(source.rglob(name))
        if not matches:
            raise FileNotFoundError(f"{name} not found in downloaded dataset at {source}")
        (root / name).write_bytes(matches[0].read_bytes())
