"""Download and preprocess MovieLens into the library's local dataset schema."""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from streamlit_recommenders.data.prepare._progress import ProgressBar
from streamlit_recommenders.data.prepare.manifest import is_complete, read_manifest, write_manifest

MOVIELENS_URLS = {
    "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "ml-latest": "https://files.grouplens.org/datasets/movielens/ml-latest.zip",
    "ml-25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
    "ml-32m": "https://files.grouplens.org/datasets/movielens/ml-32m.zip",
}


def prepare_movielens(
    dataset: str = "ml-32m",
    root: str | Path | None = None,
    *,
    with_posters: bool = False,
    tmdb_api_key: str | None = None,
    poster_limit: int = 0,
    sleep: float = 0.03,
    force: bool = False,
) -> Path:
    """Download MovieLens and write items.csv/interactions.csv (+ optional posters).

    Idempotent: skips work when the folder is already marked complete unless
    ``force`` is set or posters are newly requested. Returns the dataset folder.

    Args:
        dataset: MovieLens release key from ``MOVIELENS_URLS``.
        root: Output folder; defaults to ``data/<dataset>``.
        with_posters: Fetch TMDB descriptions and posters after building items.
        tmdb_api_key: TMDB API key; falls back to ``TmdbClient.from_env`` (which
            reads ``TMDB_API_KEY``/``TMDB_BEARER_TOKEN``) when not given.
        poster_limit: Max posters to fetch (0 = all matched items).
        sleep: Seconds to sleep between TMDB requests.
        force: Rebuild even if the folder is already marked complete.

    Returns:
        Path to the prepared dataset folder.

    Raises:
        ValueError: If ``dataset`` is not a known MovieLens key.
    """
    if dataset not in MOVIELENS_URLS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {sorted(MOVIELENS_URLS)}")

    root = Path(root) if root else Path("data") / dataset
    root.mkdir(parents=True, exist_ok=True)

    if is_complete(root) and not force:
        manifest = read_manifest(root)
        if manifest.get("with_posters", False) or not with_posters:
            print(f"{root} already prepared (complete). Use force=True to rebuild.")
            return root

    raw_dir = root / "raw"
    zip_path = root / f"{dataset}.zip"
    _ensure_movielens(dataset, raw_dir, zip_path, force=force)

    movies = _read_required_csv(raw_dir, "movies.csv")
    ratings = _read_required_csv(raw_dir, "ratings.csv")
    links = _read_optional_csv(raw_dir, "links.csv")

    items = build_items(movies, links)
    interactions = build_interactions(ratings)

    if with_posters:
        from streamlit_recommenders.data.prepare.tmdb import TmdbClient, enrich_from_tmdb

        client = TmdbClient(api_key=tmdb_api_key) if tmdb_api_key else TmdbClient.from_env()
        items, _ = enrich_from_tmdb(
            items,
            client,
            data_dir=root,
            download_posters=True,
            poster_limit=poster_limit,
            sleep_seconds=sleep,
            force=force,
        )

    items.to_csv(root / "items.csv", index=False)
    interactions.to_csv(root / "interactions.csv", index=False)
    write_manifest(
        root,
        name=dataset,
        n_items=int(len(items)),
        n_interactions=int(len(interactions)),
        with_posters=bool(with_posters),
    )
    print(f"Prepared {len(items):,} items and {len(interactions):,} interactions in {root}")
    return root


def build_items(movies: pd.DataFrame, links: pd.DataFrame | None) -> pd.DataFrame:
    """Build the items table from MovieLens movies and optional links.

    Renames ``movieId`` to ``item_id``, extracts the release year from the
    title, and merges IMDb/TMDB ids from ``links.csv`` when available.

    Args:
        movies: Parsed ``movies.csv`` (needs ``movieId`` and ``title``).
        links: Parsed ``links.csv`` (needs ``movieId``), or None.

    Returns:
        Items DataFrame with the available columns among ``item_id``, ``title``,
        ``genres``, ``year``, ``imdb_id``, ``tmdb_id``.

    Raises:
        ValueError: If required columns are missing.
    """
    require_columns(movies, ["movieId", "title"], "movies.csv")
    items = movies.rename(columns={"movieId": "item_id"}).copy()
    if "genres" not in items.columns:
        items["genres"] = ""
    items["year"] = items["title"].map(_extract_year)

    if links is not None:
        require_columns(links, ["movieId"], "links.csv")
        link_cols = [col for col in ["movieId", "imdbId", "tmdbId"] if col in links.columns]
        links = links[link_cols].rename(
            columns={"movieId": "item_id", "imdbId": "imdb_id", "tmdbId": "tmdb_id"}
        )
        items = items.merge(links, on="item_id", how="left")
        if "imdb_id" in items.columns:
            items["imdb_id"] = items["imdb_id"].map(_format_imdb_id)

    ordered = ["item_id", "title", "genres", "year", "imdb_id", "tmdb_id"]
    return items[[column for column in ordered if column in items.columns]]


def build_interactions(ratings: pd.DataFrame) -> pd.DataFrame:
    """Build the interactions table from MovieLens ratings.

    Renames ``userId``/``movieId`` to ``user_id``/``item_id`` and keeps
    ``rating`` plus ``timestamp`` when present.

    Args:
        ratings: Parsed ``ratings.csv`` (needs ``userId``, ``movieId``,
            ``rating``).

    Returns:
        Interactions DataFrame with ``user_id``, ``item_id``, ``rating`` and,
        if available, ``timestamp``.

    Raises:
        ValueError: If required columns are missing.
    """
    require_columns(ratings, ["userId", "movieId", "rating"], "ratings.csv")
    interactions = ratings.rename(columns={"userId": "user_id", "movieId": "item_id"})
    keep = ["user_id", "item_id", "rating"]
    if "timestamp" in interactions.columns:
        keep.append("timestamp")
    return interactions[keep]


def _ensure_movielens(dataset: str, raw_dir: Path, zip_path: Path, *, force: bool) -> None:
    """Ensure raw movies/ratings CSVs exist, downloading and extracting if needed."""
    if raw_dir.joinpath("movies.csv").exists() and raw_dir.joinpath("ratings.csv").exists() and not force:
        print(f"Using existing raw MovieLens files in {raw_dir}")
        return
    _download_file(MOVIELENS_URLS[dataset], zip_path, force=force)
    _extract_movielens(zip_path, raw_dir)


def _download_file(url: str, output_path: Path, *, force: bool) -> None:
    """Stream ``url`` to ``output_path`` with a progress bar, skipping if present."""
    if output_path.exists() and not force:
        print(f"Using existing download {output_path}")
        return
    print(f"Downloading {url}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, output_path.open("wb") as output:
        total = int(response.headers.get("Content-Length", "0"))
        bar = ProgressBar(total, label=output_path.name)
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            downloaded += len(chunk)
            bar.update(downloaded)
        bar.finish()


def _extract_movielens(zip_path: Path, raw_dir: Path) -> None:
    """Extract CSV members from the MovieLens ZIP into a fresh ``raw_dir``."""
    print(f"Extracting {zip_path} to {raw_dir}")
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        members = [member for member in archive.namelist() if member.endswith(".csv")]
        bar = ProgressBar(len(members), label="extract")
        for index, member in enumerate(members, start=1):
            target = raw_dir / Path(member).name
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            bar.update(index)
        bar.finish()


def _read_required_csv(raw_dir: Path, name: str) -> pd.DataFrame:
    """Read a required CSV from ``raw_dir``, raising if it is absent."""
    path = raw_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; MovieLens ZIP did not contain {name}")
    return pd.read_csv(path)


def _read_optional_csv(raw_dir: Path, name: str) -> pd.DataFrame | None:
    """Read an optional CSV from ``raw_dir``, returning None if it is absent."""
    path = raw_dir / name
    return pd.read_csv(path) if path.exists() else None


def _extract_year(title: str) -> str:
    """Extract a trailing ``(YYYY)`` year from a title, or "" if absent."""
    text = str(title)
    if len(text) >= 6 and text.endswith(")") and text[-5:-1].isdigit():
        return text[-5:-1]
    return ""


def _format_imdb_id(value: Any) -> str:
    """Format a numeric IMDb id as a zero-padded ``tt`` string."""
    if pd.isna(value):
        return ""
    try:
        return f"tt{int(value):07d}"
    except (TypeError, ValueError):
        return str(value)


def require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    """Raise if any required column is absent from ``df``.

    Args:
        df: DataFrame to check.
        columns: Required column names.
        name: Table label used in the error message.

    Raises:
        ValueError: If one or more columns are missing.
    """
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
