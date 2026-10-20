import pandas as pd

import streamlit_recommenders as sr
from streamlit_recommenders.data.media import resolve_image_urls
from streamlit_recommenders.data.prepare import is_complete, read_manifest, write_manifest
from streamlit_recommenders.data.prepare.movielens import build_interactions, build_items
from streamlit_recommenders.data.prepare.tmdb import write_completeness_report


def test_build_items_maps_columns_and_year():
    movies = pd.DataFrame(
        {"movieId": [1, 2], "title": ["A (1999)", "B"], "genres": ["x|y", "z"]}
    )
    links = pd.DataFrame({"movieId": [1, 2], "imdbId": [111, 222], "tmdbId": [10, 20]})

    items = build_items(movies, links)

    assert list(items["item_id"]) == [1, 2]
    assert list(items["year"]) == ["1999", ""]
    assert list(items["imdb_id"]) == ["tt0000111", "tt0000222"]
    assert list(items["tmdb_id"]) == [10, 20]


def test_build_items_without_links():
    movies = pd.DataFrame({"movieId": [1], "title": ["A (2000)"]})
    items = build_items(movies, None)
    assert "genres" in items.columns
    assert "tmdb_id" not in items.columns


def test_build_interactions_keeps_timestamp():
    ratings = pd.DataFrame(
        {"userId": [1], "movieId": [1], "rating": [4.0], "timestamp": [123]}
    )
    interactions = build_interactions(ratings)
    assert list(interactions.columns) == ["user_id", "item_id", "rating", "timestamp"]


def test_manifest_roundtrip_and_completeness(tmp_path):
    assert is_complete(tmp_path) is False
    write_manifest(tmp_path, name="ml-test", n_items=3, n_interactions=5, with_posters=False)
    assert is_complete(tmp_path) is True
    manifest = read_manifest(tmp_path)
    assert manifest["name"] == "ml-test"
    assert manifest["with_posters"] is False


def test_resolve_image_urls_falls_back_to_placeholder(tmp_path):
    poster = tmp_path / "posters" / "10.jpg"
    poster.parent.mkdir(parents=True)
    poster.write_bytes(b"fake")
    items = pd.DataFrame(
        {"item_id": [1, 2], "poster_path": ["posters/10.jpg", "posters/missing.jpg"]}
    )

    resolved = resolve_image_urls(items, tmp_path)

    assert resolved.iloc[0] == str(poster)
    assert resolved.iloc[1] != str(poster)  # placeholder fallback


def test_resolve_image_urls_passes_remote_urls_through(tmp_path):
    items = pd.DataFrame(
        {"item_id": [1], "image_url": ["https://example.com/cover.jpg"]}
    )

    resolved = resolve_image_urls(items, tmp_path)

    assert resolved.iloc[0] == "https://example.com/cover.jpg"


def test_resolve_image_urls_falls_through_broken_image_url(tmp_path):
    # image_url written relative to a different working directory must not
    # shadow a valid root-relative poster_path.
    poster = tmp_path / "posters" / "10.jpg"
    poster.parent.mkdir(parents=True)
    poster.write_bytes(b"fake")
    items = pd.DataFrame(
        {
            "item_id": [1],
            "image_url": ["data/some-dataset/posters/10.jpg"],
            "poster_path": ["posters/10.jpg"],
        }
    )

    resolved = resolve_image_urls(items, tmp_path)

    assert resolved.iloc[0] == str(poster)


def test_load_dataset(tmp_path):
    poster = tmp_path / "posters" / "10.jpg"
    poster.parent.mkdir(parents=True)
    poster.write_bytes(b"fake")
    pd.DataFrame(
        {"item_id": [1, 2], "title": ["A", "B"], "poster_path": ["posters/10.jpg", ""]}
    ).to_csv(tmp_path / "items.csv", index=False)
    pd.DataFrame({"user_id": [1], "item_id": [1]}).to_csv(
        tmp_path / "train_interactions.csv", index=False
    )
    pd.DataFrame({"user_id": [1], "item_id": [2]}).to_csv(
        tmp_path / "test_interactions.csv", index=False
    )

    items, train, test = sr.load_dataset(tmp_path)

    assert "image_url" in items.columns
    assert items.loc[items["item_id"] == 1, "image_url"].iloc[0] == str(poster)
    assert train is not None and len(train) == 1
    assert test is not None and len(test) == 1


def test_completeness_report_counts_gaps(tmp_path):
    poster = tmp_path / "posters" / "10.jpg"
    poster.parent.mkdir(parents=True)
    poster.write_bytes(b"fake")
    items = pd.DataFrame(
        {
            "item_id": [1, 2],
            "poster_path": ["posters/10.jpg", ""],
            "description": ["plot", ""],
        }
    )

    report = write_completeness_report(items, tmp_path)

    assert (tmp_path / "metadata_completeness.csv").exists()
    assert int(report["missing_poster"].sum()) == 1
    assert int(report["missing_description"].sum()) == 1


class _FakeResponse:
    def __init__(self, chunks, fail_after=None):
        self.headers = {"Content-Length": str(sum(len(c) for c in chunks))}
        self._chunks = list(chunks)
        self._fail_after = fail_after
        self._reads = 0

    def read(self, _size=-1):
        if self._fail_after is not None and self._reads >= self._fail_after:
            raise TimeoutError("read timed out")
        self._reads += 1
        return self._chunks.pop(0) if self._chunks else b""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_download_file_uses_certifi_context(tmp_path, monkeypatch):
    import ssl
    import urllib.request

    from streamlit_recommenders.data.prepare import movielens

    captured = {}

    def fake_urlopen(url, *, timeout, context):
        captured["context"] = context
        return _FakeResponse([b"abc"])

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    target = tmp_path / "ds.zip"
    movielens._download_file("https://example.com/ds.zip", target, force=False)

    assert target.read_bytes() == b"abc"
    assert isinstance(captured["context"], ssl.SSLContext)
    assert captured["context"].verify_mode == ssl.CERT_REQUIRED


def test_interrupted_download_leaves_no_reusable_file(tmp_path, monkeypatch):
    import urllib.request

    import pytest

    from streamlit_recommenders.data.prepare import movielens

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda url, **kw: _FakeResponse([b"a", b"b"], fail_after=1),
    )
    target = tmp_path / "ds.zip"
    with pytest.raises(TimeoutError):
        movielens._download_file("https://example.com/ds.zip", target, force=False)

    # A rerun must re-download instead of trusting a truncated archive.
    assert not target.exists()
    assert not (tmp_path / "ds.zip.part").exists()


def test_tmdb_read_timeout_is_retried_not_fatal(monkeypatch):
    import urllib.request

    from streamlit_recommenders.data.prepare import tmdb

    monkeypatch.setattr(tmdb.time, "sleep", lambda _s: None)

    def fake_urlopen(request, **kw):
        return _FakeResponse([b"{}"], fail_after=0)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert tmdb.TmdbClient(api_key="k").movie(1, attempts=2) == {}


def test_tmdb_rejected_credentials_stop_enrichment(monkeypatch):
    import io
    import urllib.error
    import urllib.request

    import pytest

    from streamlit_recommenders.data.prepare import tmdb

    def fake_urlopen(request, **kw):
        raise urllib.error.HTTPError("u", 401, "Unauthorized", {}, io.BytesIO())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(SystemExit, match="401"):
        tmdb.TmdbClient(api_key="bad").movie(1)


def test_cli_reads_dotenv_from_working_directory(tmp_path, monkeypatch):
    # Installed from PyPI the package lives in site-packages, so the .env must
    # be looked up from where the user runs the command, not next to the code.
    import sys

    import pytest

    from streamlit_recommenders.data.prepare import __main__ as cli

    (tmp_path / ".env").write_text("TMDB_API_KEY=from_dotenv\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TMDB_API_KEY", "placeholder")
    monkeypatch.delenv("TMDB_API_KEY")  # restored to the original on teardown
    monkeypatch.setattr(sys, "argv", ["prepare", "--dataset", "unknown"])

    with pytest.raises(ValueError):
        cli.main()

    import os

    assert os.environ.get("TMDB_API_KEY") == "from_dotenv"
