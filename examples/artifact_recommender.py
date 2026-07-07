from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit_recommenders as sr

DEFAULT_DATA_DIR = Path("data/ml-32m-filtered")
REPO_PLACEHOLDER = Path(__file__).resolve().parents[1] / "data" / "static" / "img" / "poster_not_available.png"


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
    root = Path(root_str)
    items = sr.load_items(str(root / "items.csv"))
    items["image_url"] = resolve_image_urls(items, root)

    train = sr.load_interactions(str(root / "train_interactions.csv"))
    test_path = root / "test_interactions.csv"
    test = sr.load_interactions(str(test_path)) if test_path.exists() else None
    return items, train, test


def resolve_image_urls(items: pd.DataFrame, root: Path) -> pd.Series:
    placeholder = dataset_placeholder(root)

    def resolve(value) -> str:
        if pd.isna(value) or not str(value).strip():
            return str(placeholder)
        path = Path(str(value))
        if not path.is_absolute():
            path = root / path
        return str(path) if path.exists() else str(placeholder)

    if "image_url" in items.columns:
        return items["image_url"].map(resolve)
    if "poster_path" in items.columns:
        return items["poster_path"].map(resolve)
    return pd.Series([str(placeholder)] * len(items), index=items.index)


def dataset_placeholder(root: Path) -> Path:
    local = root / "static" / "img" / "poster_not_available.png"
    if local.exists():
        return local
    return REPO_PLACEHOLDER


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


class ArtifactRecommender:
    """Adapter that exposes exported NumPy artifacts through the library contract."""

    SCORE_PARAM_NAMES = {"history_window"}

    def __init__(
        self,
        artifact_path: Path,
        interactions: pd.DataFrame,
    ) -> None:
        artifact = np.load(artifact_path, allow_pickle=False)
        self.name = artifact_path.stem
        self.model_type = str(artifact["model_type"][0])
        self.item_ids = artifact["item_ids"].tolist()
        self.item_index = {item_id: idx for idx, item_id in enumerate(self.item_ids)}
        self.weights = artifact["weights"]
        self.popularity = artifact["popularity"]
        self.interactions = interactions
        self.user_history = self._build_user_history(interactions)
        self.user_seen = {user_id: set(items) for user_id, items in self.user_history.items()}

    def get_recommendations(
        self,
        user_id: str | int,
        k: int,
        session_items: list | None = None,
        **params,
    ) -> list:
        seen = set(self.user_seen.get(user_id, set()))
        seen.update(session_items or [])
        scores = self.scores(
            user_id,
            session_items=session_items,
            **self._score_params(params),
        )
        ranked = np.argsort(scores)[::-1]
        recs = []
        for idx in ranked:
            item_id = self.item_ids[int(idx)]
            if item_id in seen:
                continue
            recs.append(item_id)
            if len(recs) >= k:
                break
        return recs

    def scores(
        self,
        user_id: str | int,
        session_items: list | None = None,
        history_window: int | str | None = None,
    ) -> np.ndarray:
        session_items = session_items or []
        if self.model_type == "sequential_cf":
            anchor = self._last_item(user_id, session_items)
            if anchor in self.item_index:
                scores = self.weights[self.item_index[anchor]].copy()
                if scores.any():
                    return scores
            return self.popularity.copy()

        vector = self._user_vector(user_id, session_items, history_window=history_window)
        if not vector.any():
            return self.popularity.copy()
        return vector @ self.weights

    def score_frame(
        self,
        user_id: str | int,
        session_items: list | None = None,
        items: pd.DataFrame | None = None,
        **params,
    ) -> pd.DataFrame:
        scores = self.scores(
            user_id,
            session_items=session_items,
            **self._score_params(params),
        )
        frame = pd.DataFrame({"item_id": self.item_ids, "score": scores})
        if items is not None and {"item_id", "title"}.issubset(items.columns):
            frame = frame.merge(items[["item_id", "title"]], on="item_id", how="left")
        return frame.sort_values("score", ascending=False)

    def _user_vector(
        self,
        user_id: str | int,
        session_items: list,
        *,
        history_window: int | str | None = None,
    ) -> np.ndarray:
        vector = np.zeros(len(self.item_ids), dtype=np.float32)
        history = self.user_history.get(user_id, []) + list(session_items)
        if history_window not in (None, "All"):
            history = history[-int(history_window) :]
        for item_id in history:
            if item_id in self.item_index:
                vector[self.item_index[item_id]] = 1.0
        return vector

    def _last_item(self, user_id: str | int, session_items: list) -> str | int | None:
        if session_items:
            return session_items[-1]
        history = self.user_history.get(user_id, [])
        if not history:
            return None
        return history[-1]

    def _score_params(self, params: dict) -> dict:
        return {name: params[name] for name in self.SCORE_PARAM_NAMES if name in params}

    @staticmethod
    def _build_user_history(interactions: pd.DataFrame) -> dict:
        frame = interactions
        if "timestamp" in frame.columns:
            frame = frame.sort_values(["user_id", "timestamp"])
        return frame.groupby("user_id")["item_id"].agg(list).to_dict()
