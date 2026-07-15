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
