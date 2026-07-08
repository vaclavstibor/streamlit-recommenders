"""Download and preprocess MovieLens into the local streamlit-recommenders schema.

Examples:
  python scripts/preprocess_movielens.py --dataset ml-32m
  TMDB_API_KEY=... python scripts/preprocess_movielens.py --dataset ml-25m --with-tmdb --download-posters

The generated data stays under data/<dataset-name>/, which is gitignored.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


MOVIELENS_URLS = {
    "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "ml-latest": "https://files.grouplens.org/datasets/movielens/ml-latest.zip",
    "ml-25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
    "ml-32m": "https://files.grouplens.org/datasets/movielens/ml-32m.zip",
}

TMDB_API_URL = "https://api.themoviedb.org/3/movie/{tmdb_id}"
TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w342{poster_path}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=sorted(MOVIELENS_URLS),
        default="ml-32m",
        help="MovieLens dataset to download.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Defaults to data/<dataset>.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download and extract the source ZIP even when raw CSVs already exist.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite generated CSVs and downloaded posters.",
    )
    parser.add_argument(
        "--with-tmdb",
        action="store_true",
        help="Fetch TMDB metadata using TMDB_API_KEY or TMDB_BEARER_TOKEN.",
    )
    parser.add_argument(
        "--download-posters",
        action="store_true",
        help="Download poster files from TMDB. Implies --with-tmdb.",
    )
    parser.add_argument(
        "--poster-limit",
        type=int,
        default=0,
        help="Maximum number of posters to download. Use 0 for all matched movies.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.03,
        help="Delay between TMDB requests to be polite to the API.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.output or Path("data") / args.dataset
    raw_dir = data_dir / "raw"
    zip_path = data_dir / f"{args.dataset}.zip"

    data_dir.mkdir(parents=True, exist_ok=True)
    ensure_movielens(args.dataset, raw_dir, zip_path, force=args.force_download)

    movies = read_required_csv(raw_dir, "movies.csv")
    ratings = read_required_csv(raw_dir, "ratings.csv")
    links = read_optional_csv(raw_dir, "links.csv")

    items = build_items(movies, links)
    interactions = build_interactions(ratings)

    if args.with_tmdb or args.download_posters:
        tmdb_client = TmdbClient.from_env()
        items = enrich_from_tmdb(
            items,
            tmdb_client,
            data_dir=data_dir,
            download_posters=args.download_posters,
            poster_limit=args.poster_limit,
            sleep_seconds=args.sleep,
            force=args.force,
        )

    write_csv(items, data_dir / "items.csv", force=args.force)
    write_csv(interactions, data_dir / "interactions.csv", force=args.force)
    print(f"Prepared {len(items):,} items and {len(interactions):,} interactions in {data_dir}")
    print("Next: python examples/train_baseline_artifacts.py --data " + str(data_dir))


def ensure_movielens(dataset: str, raw_dir: Path, zip_path: Path, *, force: bool) -> None:
    if raw_dir.joinpath("movies.csv").exists() and raw_dir.joinpath("ratings.csv").exists() and not force:
        print(f"Using existing raw MovieLens files in {raw_dir}")
        return

    url = MOVIELENS_URLS[dataset]
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    download_file(url, zip_path, force=force)
    extract_movielens(zip_path, raw_dir)


def download_file(url: str, output_path: Path, *, force: bool) -> None:
    if output_path.exists() and not force:
        print(f"Using existing download {output_path}")
        return

    print(f"Downloading {url}")
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


def extract_movielens(zip_path: Path, raw_dir: Path) -> None:
    print(f"Extracting {zip_path} to {raw_dir}")
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        members = [member for member in archive.namelist() if member.endswith(".csv")]
        bar = ProgressBar(len(members), label="extract")
        for index, member in enumerate(members, start=1):
            target_name = Path(member).name
            with archive.open(member) as source, raw_dir.joinpath(target_name).open("wb") as target:
                shutil.copyfileobj(source, target)
            bar.update(index)
        bar.finish()


def read_required_csv(raw_dir: Path, name: str) -> pd.DataFrame:
    path = raw_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; MovieLens ZIP did not contain {name}")
    return pd.read_csv(path)


def read_optional_csv(raw_dir: Path, name: str) -> pd.DataFrame | None:
    path = raw_dir / name
    return pd.read_csv(path) if path.exists() else None


def build_items(movies: pd.DataFrame, links: pd.DataFrame | None) -> pd.DataFrame:
    require_columns(movies, ["movieId", "title"], "movies.csv")
    items = movies.rename(columns={"movieId": "item_id"}).copy()
    if "genres" not in items.columns:
        items["genres"] = ""

    items["year"] = items["title"].map(extract_year)
    if links is not None:
        require_columns(links, ["movieId"], "links.csv")
        link_cols = [col for col in ["movieId", "imdbId", "tmdbId"] if col in links.columns]
        links = links[link_cols].rename(
            columns={"movieId": "item_id", "imdbId": "imdb_id", "tmdbId": "tmdb_id"}
        )
        items = items.merge(links, on="item_id", how="left")
        if "imdb_id" in items.columns:
            items["imdb_id"] = items["imdb_id"].map(format_imdb_id)

    ordered = ["item_id", "title", "genres", "year", "imdb_id", "tmdb_id"]
    return items[[column for column in ordered if column in items.columns]]


def build_interactions(ratings: pd.DataFrame) -> pd.DataFrame:
    require_columns(ratings, ["userId", "movieId", "rating"], "ratings.csv")
    interactions = ratings.rename(columns={"userId": "user_id", "movieId": "item_id"})
    keep = ["user_id", "item_id", "rating"]
    if "timestamp" in interactions.columns:
        keep.append("timestamp")
    return interactions[keep]


def enrich_from_tmdb(
    items: pd.DataFrame,
    client: "TmdbClient",
    *,
    data_dir: Path,
    download_posters: bool,
    poster_limit: int,
    sleep_seconds: float,
    force: bool,
) -> pd.DataFrame:
    require_columns(items, ["tmdb_id"], "items")
    enriched = items.copy()
    posters_dir = data_dir / "posters"
    if download_posters:
        posters_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows: list[dict[str, Any]] = []
    candidates = enriched.dropna(subset=["tmdb_id"])
    if poster_limit > 0:
        candidates = candidates.head(poster_limit)

    bar = ProgressBar(len(candidates), label="tmdb")
    for index, (_, row) in enumerate(candidates.iterrows(), start=1):
        tmdb_id = int(row["tmdb_id"])
        try:
            metadata = client.movie(tmdb_id)
        except urllib.error.HTTPError as exc:
            print(f"\nTMDB skipped {tmdb_id}: HTTP {exc.code}", file=sys.stderr)
            metadata = {}
        except urllib.error.URLError as exc:
            print(f"\nTMDB skipped {tmdb_id}: {exc.reason}", file=sys.stderr)
            metadata = {}

        poster_path = metadata.get("poster_path")
        local_poster = ""
        if download_posters and poster_path:
            local_poster = download_poster(
                poster_path,
                posters_dir / f"{tmdb_id}.jpg",
                force=force,
            )
        local_poster_path = Path(local_poster).resolve() if local_poster else None

        metadata_rows.append(
            {
                "item_id": row["item_id"],
                "description": metadata.get("overview", ""),
                "release_date": metadata.get("release_date", ""),
                "popularity": metadata.get("popularity", ""),
                "tmdb_poster_path": poster_path or "",
                "poster_path": relative_to(data_dir.resolve(), str(local_poster_path))
                if local_poster_path
                else "",
                "image_url": str(local_poster_path) if local_poster_path else "",
            }
        )
        bar.update(index)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    bar.finish()

    if not metadata_rows:
        return enriched
    metadata_df = pd.DataFrame(metadata_rows)
    return enriched.merge(metadata_df, on="item_id", how="left")


def download_poster(poster_path: str, output_path: Path, *, force: bool) -> str:
    if output_path.exists() and not force:
        return str(output_path)
    url = TMDB_IMAGE_URL.format(poster_path=poster_path)
    with urllib.request.urlopen(url) as response, output_path.open("wb") as output:
        shutil.copyfileobj(response, output)
    return str(output_path)


def write_csv(df: pd.DataFrame, path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists. Re-run with --force to overwrite.")
    df.to_csv(path, index=False)
    print(f"Wrote {path}")


def extract_year(title: str) -> str:
    text = str(title)
    if len(text) >= 6 and text.endswith(")") and text[-5:-1].isdigit():
        return text[-5:-1]
    return ""


def format_imdb_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    try:
        return f"tt{int(value):07d}"
    except (TypeError, ValueError):
        return str(value)


def relative_to(root: Path, path: str) -> str:
    if not path:
        return ""
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return path


def require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


class TmdbClient:
    def __init__(self, *, api_key: str = "", bearer_token: str = "") -> None:
        self.api_key = api_key
        self.bearer_token = bearer_token

    @classmethod
    def from_env(cls) -> "TmdbClient":
        client = cls(
            api_key=os.environ.get("TMDB_API_KEY", ""),
            bearer_token=os.environ.get("TMDB_BEARER_TOKEN", ""),
        )
        if not client.api_key and not client.bearer_token:
            raise SystemExit("Set TMDB_API_KEY or TMDB_BEARER_TOKEN before using --with-tmdb.")
        return client

    def movie(self, tmdb_id: int) -> dict[str, Any]:
        params = {"language": "en-US"}
        if self.api_key:
            params["api_key"] = self.api_key
        url = TMDB_API_URL.format(tmdb_id=tmdb_id) + "?" + urllib.parse.urlencode(params)
        request = urllib.request.Request(url, headers=self._headers())
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))

    def _headers(self) -> dict[str, str]:
        if not self.bearer_token:
            return {}
        return {"Authorization": f"Bearer {self.bearer_token}"}


class ProgressBar:
    def __init__(self, total: int, *, label: str, width: int = 28) -> None:
        self.total = max(total, 0)
        self.label = label
        self.width = width
        self.last_text = ""

    def update(self, current: int) -> None:
        if self.total <= 0:
            text = f"{self.label}: {current:,}"
        else:
            ratio = min(max(current / self.total, 0.0), 1.0)
            filled = int(self.width * ratio)
            bar = "#" * filled + "-" * (self.width - filled)
            text = f"{self.label}: [{bar}] {current:,}/{self.total:,} ({ratio:.0%})"
        if text != self.last_text:
            print("\r" + text, end="", flush=True)
            self.last_text = text

    def finish(self) -> None:
        if self.last_text:
            print()


if __name__ == "__main__":
    main()
