from pathlib import Path

import pandas as pd

import streamlit_recommenders as sr
from streamlit_recommenders.runtime.seen import SESSION_USER_ID

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

ArtifactRecommender = sr.ArtifactRecommender


def test_polished_examples_exist():
    assert (EXAMPLES / "reference_recommenders.py").exists()
    assert (EXAMPLES / "compare_models_rows.py").exists()
    assert (EXAMPLES / "swipe_deck_cards.py").exists()
    assert (EXAMPLES / "compare_models_grid.py").exists()
    assert (EXAMPLES / "train_baseline_artifacts.py").exists()


def test_artifact_loading_is_public_library_api():
    # Loading exported artifacts is core library API, not example glue.
    assert callable(sr.load_artifacts)
    assert callable(sr.load_local_dataset)


def test_artifact_recommender_is_library_class():
    assert issubclass(sr.ArtifactRecommender, sr.BaseRecommender)


def test_legacy_examples_removed():
    assert not (EXAMPLES / "builtin_recommenders.py").exists()
    # Its glue folded into sr.load_local_dataset / sr.load_artifacts.
    assert not (EXAMPLES / "artifact_recommender.py").exists()
    assert not (EXAMPLES / "_bootstrap.py").exists()
    assert not (EXAMPLES / "minimal_demo.py").exists()
    assert not (EXAMPLES / "pickle_demo.py").exists()
    assert not (EXAMPLES / "matrix_demo.py").exists()
    assert not (EXAMPLES / "showcase_demo.py").exists()
    assert not (EXAMPLES / "baseline_comparison_demo.py").exists()
    assert not (EXAMPLES / "appendix_demo.py").exists()
    assert not (EXAMPLES / "appendix.md").exists()
    assert not (EXAMPLES / "sequence_cf_demo.py").exists()
    assert not (EXAMPLES / "generate_sample_data.py").exists()
    assert not (EXAMPLES / "sample_data").exists()


def test_artifact_recommender_handles_session_user(tmp_path):
    artifact = tmp_path / "itemknn.npz"
    write_artifact(artifact, model_type="itemknn")
    interactions = pd.DataFrame({"user_id": [1], "item_id": [10]})
    model = ArtifactRecommender(artifact, interactions)

    recs = model.get_recommendations(SESSION_USER_ID, 2, session_items=[10])

    assert recs == [12, 11]


def test_artifact_recommender_handles_existing_user(tmp_path):
    artifact = tmp_path / "sequential_cf.npz"
    write_artifact(artifact, model_type="sequential_cf")
    interactions = pd.DataFrame(
        {"user_id": [1, 1], "item_id": [10, 11], "timestamp": [1, 2]}
    )
    model = ArtifactRecommender(artifact, interactions)

    recs = model.get_recommendations(1, 2)

    assert recs == [12]


def test_artifact_recommender_ignores_framework_context_params(tmp_path):
    artifact = tmp_path / "itemknn.npz"
    write_artifact(artifact, model_type="itemknn")
    interactions = pd.DataFrame({"user_id": [1], "item_id": [10]})
    model = ArtifactRecommender(artifact, interactions)

    recs = model.get_recommendations(
        1,
        2,
        session_items=[10],
        selections=[{"section": "External ItemKNN", "item_id": 10}],
        ignored_param=1.0,
    )

    assert recs == [12, 11]


def write_artifact(path: Path, *, model_type: str) -> None:
    import numpy as np

    weights = np.array(
        [
            [0.0, 0.2, 0.8],
            [0.1, 0.0, 0.9],
            [0.7, 0.3, 0.0],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(
        path,
        model_type=np.array([model_type]),
        item_ids=np.array([10, 11, 12]),
        weights=weights,
        popularity=np.array([3.0, 2.0, 1.0], dtype=np.float32),
    )
