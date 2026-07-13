from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit_recommenders as sr
from streamlit_recommenders import ArtifactRecommender

__all__ = [
    "ArtifactRecommender",
    "data_dir",
    "load_artifact_dataset",
    "load_artifact_models",
]

DEFAULT_DATA_DIR = Path("data/ml-latest-small")


def data_dir() -> Path:
    return Path(os.environ.get("SR_DATA_DIR", DEFAULT_DATA_DIR))


def load_artifact_dataset(
    data: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    root = data or data_dir()
    return _load_artifact_dataset_cached(str(root))


@st.cache_resource(show_spinner=False)
def _load_artifact_dataset_cached(
    root_str: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    return sr.load_local_dataset(root_str)


def load_artifact_models(
    interactions: pd.DataFrame,
    data: Path | None = None,
    *,
    prefix: str = "",
) -> dict[str, ArtifactRecommender]:
    root = data or data_dir()
    return _load_artifact_models_cached(str(root), prefix, interactions)


@st.cache_resource(show_spinner=False)
def _load_artifact_models_cached(
    root_str: str,
    prefix: str,
    _interactions: pd.DataFrame,
) -> dict[str, ArtifactRecommender]:
    root = Path(root_str)
    artifacts = root / "artifacts"
    paths = {
        f"{prefix}ItemKNN": artifacts / "itemknn.npz",
        f"{prefix}EASE": artifacts / "ease.npz",
        f"{prefix}Sequential CF": artifacts / "sequential_cf.npz",
    }
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Missing trained artifacts. Run:\n"
            f"  python examples/train_baseline_artifacts.py --data {root}\n\n"
            f"Missing:\n{missing_text}"
        )
    return {name: ArtifactRecommender(path, _interactions) for name, path in paths.items()}
