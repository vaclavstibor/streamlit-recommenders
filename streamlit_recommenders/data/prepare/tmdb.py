"""Robust TMDB enrichment: metadata + posters with retry/backoff and a completeness report."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from streamlit_recommenders.data.prepare._progress import ProgressBar

TMDB_API_URL = "https://api.themoviedb.org/3/movie/{tmdb_id}"
TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w342{poster_path}"
COMPLETENESS_FILE = "metadata_completeness.csv"


class TmdbClient:
    """Small urllib TMDB client with retry, backoff, and 429 rate-limit handling."""

    def __init__(self, *, api_key: str = "", bearer_token: str = "") -> None:
        """Store TMDB credentials.

        Args:
            api_key: TMDB v3 API key, sent as a query parameter.
            bearer_token: TMDB v4 bearer token, sent as an Authorization header.
        """
        self.api_key = api_key
        self.bearer_token = bearer_token

    @classmethod
    def from_env(cls) -> TmdbClient:
        """Build a client from ``TMDB_API_KEY``/``TMDB_BEARER_TOKEN``.

        Returns:
            A configured TmdbClient.

        Raises:
            SystemExit: If neither environment variable is set.
        """
        client = cls(
            api_key=os.environ.get("TMDB_API_KEY", ""),
            bearer_token=os.environ.get("TMDB_BEARER_TOKEN", ""),
        )
        if not client.api_key and not client.bearer_token:
            raise SystemExit("Set TMDB_API_KEY or TMDB_BEARER_TOKEN to fetch posters.")
        return client

    def movie(self, tmdb_id: int, *, attempts: int = 4) -> dict[str, Any]:
        """Fetch a movie's TMDB metadata, retrying on rate limits and errors.

        Args:
            tmdb_id: TMDB movie id.
            attempts: Maximum number of request attempts.

        Returns:
            The parsed JSON metadata, or an empty dict on failure.
        """
        params = {"language": "en-US"}
        if self.api_key:
            params["api_key"] = self.api_key
        url = TMDB_API_URL.format(tmdb_id=tmdb_id) + "?" + urllib.parse.urlencode(params)

        for attempt in range(1, attempts + 1):
            try:
                request = urllib.request.Request(url, headers=self._headers())
                with urllib.request.urlopen(request, timeout=20) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    time.sleep(min(2 * attempt, 10))
                    continue
                return {}
            except urllib.error.URLError:
                if attempt == attempts:
                    return {}
                time.sleep(1.5 * attempt)
        return {}

    def _headers(self) -> dict[str, str]:
        """Return the Authorization header for bearer auth, or empty when unset."""
        if not self.bearer_token:
            return {}
        return {"Authorization": f"Bearer {self.bearer_token}"}


def enrich_from_tmdb(
    items: pd.DataFrame,
    client: TmdbClient,
    *,
    data_dir: Path,
    download_posters: bool = True,
    poster_limit: int = 0,
    sleep_seconds: float = 0.03,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch descriptions + posters for items that carry a ``tmdb_id``.

    Returns the enriched items and a completeness report of remaining gaps.

    Args:
        items: Item table; must include a ``tmdb_id`` column.
        client: TMDB client used to fetch metadata.
        data_dir: Dataset root; posters are saved under ``data_dir/posters``.
        download_posters: Download poster images when True.
        poster_limit: Max items to process (0 = all with a ``tmdb_id``).
        sleep_seconds: Seconds to sleep between requests.
        force: Re-download posters even if already present.

    Returns:
        Tuple of ``(enriched_items, completeness_report)``.

    Raises:
        ValueError: If ``items`` lacks a ``tmdb_id`` column.
    """
    if "tmdb_id" not in items.columns:
        raise ValueError("items must include a tmdb_id column for TMDB enrichment")

    enriched = items.copy()
    posters_dir = data_dir / "posters"
    if download_posters:
        posters_dir.mkdir(parents=True, exist_ok=True)

    candidates = enriched.dropna(subset=["tmdb_id"])
    if poster_limit > 0:
        candidates = candidates.head(poster_limit)

    rows: list[dict[str, Any]] = []
    bar = ProgressBar(len(candidates), label="tmdb")
    for index, (_, row) in enumerate(candidates.iterrows(), start=1):
        tmdb_id = int(row["tmdb_id"])
        metadata = client.movie(tmdb_id)
        poster_path = metadata.get("poster_path")
        local_poster = ""
        if download_posters and poster_path:
            local_poster = _download_poster(poster_path, posters_dir / f"{tmdb_id}.jpg", force=force)

        rows.append(
            {
                "item_id": row["item_id"],
                "description": metadata.get("overview", "") or "",
                "release_date": metadata.get("release_date", "") or "",
                # Root-relative so the dataset folder stays portable; the
                # loader resolves it against the dataset root at load time.
                "poster_path": _relative_to(data_dir, local_poster) if local_poster else "",
            }
        )
        bar.update(index)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    bar.finish()

    if rows:
        enriched = enriched.merge(pd.DataFrame(rows), on="item_id", how="left")

    report = write_completeness_report(enriched, data_dir)
    return enriched, report


def write_completeness_report(items: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Write and summarize per-item missing poster/description flags.

    Args:
        items: Enriched items table.
        data_dir: Dataset root; the report is written to
            ``data_dir/metadata_completeness.csv``.

    Returns:
        DataFrame of items with a missing poster and/or description.
    """
    def missing_poster(row) -> int:
        """Return 1 if the row's poster file is missing or unresolved, else 0."""
        value = row.get("poster_path", "")
        if pd.isna(value) or not str(value).strip():
            return 1
        path = Path(str(value))
        if not path.is_absolute():
            path = data_dir / path
        return int(not path.exists())

    def missing_description(row) -> int:
        """Return 1 if the row's description is missing or blank, else 0."""
        value = row.get("description", "")
        return int(pd.isna(value) or not str(value).strip())

    report_rows = []
    for _, row in items.iterrows():
        mp = missing_poster(row)
        md = missing_description(row)
        if mp or md:
            report_rows.append(
                {"item_id": row["item_id"], "missing_poster": mp, "missing_description": md}
            )

    report = pd.DataFrame(report_rows, columns=["item_id", "missing_poster", "missing_description"])
    report.to_csv(data_dir / COMPLETENESS_FILE, index=False)

    total = len(items)
    print(
        f"completeness: {total:,} items | "
        f"missing poster: {int(report['missing_poster'].sum()) if not report.empty else 0:,} | "
        f"missing description: {int(report['missing_description'].sum()) if not report.empty else 0:,}"
    )
    print(f"  report: {data_dir / COMPLETENESS_FILE}")
    return report


def _download_poster(poster_path: str, output_path: Path, *, force: bool) -> str:
    """Download a TMDB poster to ``output_path``; return its path or "" on failure."""
    if output_path.exists() and not force:
        return str(output_path)
    url = TMDB_IMAGE_URL.format(poster_path=poster_path)
    try:
        with urllib.request.urlopen(url, timeout=20) as response, output_path.open("wb") as output:
            shutil.copyfileobj(response, output)
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        print(f"\nposter download failed for {output_path.name}: {exc}", file=sys.stderr)
        return ""
    return str(output_path)


def _relative_to(root: Path, path: str) -> str:
    """Return ``path`` relative to ``root``, or unchanged if not under it."""
    if not path:
        return ""
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return path
