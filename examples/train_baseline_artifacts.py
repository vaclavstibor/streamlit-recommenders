"""Train three baseline models and export pure NumPy artifacts.

The script is intentionally outside the core package runtime. It prepares a
local dataset under data/<dataset-name>/, creates train/test splits when needed,
and writes model weights that streamlit-recommenders can inspect without
depending on training code at demo runtime.

Outputs:
  artifacts/itemknn.npz
  artifacts/ease.npz
  artifacts/sequential_cf.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from streamlit_recommenders.data.prepare._progress import ProgressBar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/ml-latest-small"),
        help="Dataset directory. Supports standard CSVs or MovieLens movies.csv/ratings.csv.",
    )
    parser.add_argument("--k-neighbors", type=int, default=100)
    parser.add_argument("--ease-l2", type=float, default=500.0)
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fallback random split ratio when timestamps are missing.",
    )
    parser.add_argument(
        "--force-split",
        action="store_true",
        help="Recreate train_interactions.csv and test_interactions.csv.",
    )
    parser.add_argument(
        "--allow-large-ease",
        action="store_true",
        help="Allow EASE for item catalogs estimated above the memory warning threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scipy_sparse = import_scipy_sparse()

    data_dir = args.data
    items, interactions = load_or_prepare_dataset(data_dir)
    train, test = load_or_create_split(
        data_dir,
        interactions,
        test_ratio=args.test_ratio,
        force=args.force_split,
    )

    item_ids = items["item_id"].tolist()
    matrix = user_item_matrix(scipy_sparse, train, item_ids)
    artifacts_dir = data_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_itemknn(matrix, item_ids, artifacts_dir / "itemknn.npz", args.k_neighbors)
    train_ease(
        matrix,
        item_ids,
        artifacts_dir / "ease.npz",
        l2=args.ease_l2,
        allow_large=args.allow_large_ease,
    )
    train_sequential(train, item_ids, artifacts_dir / "sequential_cf.npz")

    print(f"Prepared {len(items):,} items, {len(train):,} train rows, {len(test):,} test rows")
    print(f"Wrote artifacts to {artifacts_dir}")


def import_scipy_sparse():
    try:
        from scipy import sparse
    except ImportError as exc:
        raise SystemExit(
            "Training artifacts needs scipy. Install it with: pip install scipy"
        ) from exc
    return sparse


def load_or_prepare_dataset(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    items_path = data_dir / "items.csv"
    interactions_path = data_dir / "interactions.csv"
    if items_path.exists() and interactions_path.exists():
        items = pd.read_csv(items_path)
        interactions = pd.read_csv(interactions_path)
    elif (data_dir / "movies.csv").exists() and (data_dir / "ratings.csv").exists():
        items, interactions = prepare_movielens(data_dir)
        items.to_csv(items_path, index=False)
        interactions.to_csv(interactions_path, index=False)
    else:
        raise FileNotFoundError(
            f"{data_dir} must contain either items.csv/interactions.csv or movies.csv/ratings.csv"
        )

    require_columns(items, ["item_id"], "items.csv")
    require_columns(interactions, ["user_id", "item_id"], "interactions.csv")
    return items, interactions


def prepare_movielens(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    movies = pd.read_csv(data_dir / "movies.csv")
    ratings = pd.read_csv(data_dir / "ratings.csv")
    require_columns(movies, ["movieId", "title"], "movies.csv")
    require_columns(ratings, ["userId", "movieId", "rating"], "ratings.csv")

    optional_movie_cols = ["genres"] if "genres" in movies.columns else []
    items = movies.rename(
        columns={
            "movieId": "item_id",
            "genres": "genres",
        }
    )[["item_id", "title", *optional_movie_cols]]
    if (data_dir / "plots.csv").exists():
        plots = pd.read_csv(data_dir / "plots.csv")
        if {"movieId", "plot"}.issubset(plots.columns):
            plots = plots.rename(columns={"movieId": "item_id", "plot": "description"})
            items = items.merge(plots[["item_id", "description"]], on="item_id", how="left")

    img_dir = data_dir / "img"
    if img_dir.exists():
        items["poster_path"] = items["item_id"].map(lambda item_id: f"img/{item_id}.jpg")

    interactions = ratings.rename(
        columns={
            "userId": "user_id",
            "movieId": "item_id",
        }
    )
    keep = ["user_id", "item_id", "rating"]
    if "timestamp" in interactions.columns:
        keep.append("timestamp")
    interactions = interactions[keep]
    return items, interactions


def load_or_create_split(
    data_dir: Path,
    interactions: pd.DataFrame,
    *,
    test_ratio: float,
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = data_dir / "train_interactions.csv"
    test_path = data_dir / "test_interactions.csv"
    if train_path.exists() and test_path.exists() and not force:
        return pd.read_csv(train_path), pd.read_csv(test_path)

    train, test = leave_last_out_split(interactions, test_ratio=test_ratio)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return train, test


def leave_last_out_split(
    interactions: pd.DataFrame,
    *,
    test_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "timestamp" in interactions.columns:
        train_parts = []
        test_parts = []
        for _, group in interactions.sort_values("timestamp").groupby("user_id", sort=False):
            if len(group) <= 1:
                train_parts.append(group)
            else:
                train_parts.append(group.iloc[:-1])
                test_parts.append(group.iloc[-1:])
        train = pd.concat(train_parts, ignore_index=True)
        test = pd.concat(test_parts, ignore_index=True) if test_parts else interactions.iloc[0:0]
        return train, test

    shuffled = interactions.sample(frac=1.0, random_state=42).reset_index(drop=True)
    cutoff = int(len(shuffled) * (1.0 - test_ratio))
    return shuffled.iloc[:cutoff].copy(), shuffled.iloc[cutoff:].copy()


def user_item_matrix(sparse, interactions: pd.DataFrame, item_ids: list):
    user_ids = sorted(interactions["user_id"].unique().tolist())
    user_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_index = {item_id: idx for idx, item_id in enumerate(item_ids)}

    rows = interactions["user_id"].map(user_index).to_numpy()
    cols = interactions["item_id"].map(item_index).to_numpy()
    valid = ~pd.isna(cols)
    values = np.ones(int(valid.sum()), dtype=np.float32)
    return sparse.csr_matrix(
        (values, (rows[valid], cols[valid].astype(int))),
        shape=(len(user_ids), len(item_ids)),
        dtype=np.float32,
    )


def train_itemknn(matrix, item_ids: list, output_path: Path, k_neighbors: int) -> None:
    progress = ProgressBar(len(item_ids), label="[1/3] ItemKNN similarity")
    cooccurrence = (matrix.T @ matrix).toarray().astype(np.float32)
    norms = np.sqrt(np.diag(cooccurrence))
    denom = np.outer(norms, norms)
    similarity = np.divide(
        cooccurrence,
        denom,
        out=np.zeros_like(cooccurrence, dtype=np.float32),
        where=denom > 0,
    )
    np.fill_diagonal(similarity, 0.0)
    keep_topk(similarity, k_neighbors, progress=progress)
    progress.finish()
    save_artifact(
        output_path,
        model_type="itemknn",
        item_ids=item_ids,
        weights=similarity,
        popularity=np.asarray(matrix.sum(axis=0)).ravel(),
    )


def train_ease(
    matrix,
    item_ids: list,
    output_path: Path,
    *,
    l2: float,
    allow_large: bool,
) -> None:
    from scipy import linalg

    n_items = matrix.shape[1]
    estimated_gb = (n_items * n_items * 4 * 4) / (1024**3)
    if estimated_gb > 6 and not allow_large:
        raise MemoryError(
            f"EASE for {n_items:,} items may need about {estimated_gb:.1f} GB. "
            "Use --allow-large-ease if this machine can handle it."
        )

    progress = ProgressBar(3, label="[2/3] EASE closed-form solve")
    # Memory is the constraint here, not precision: the exported weights are
    # float32 anyway and gram + l2*I is symmetric positive definite and well
    # conditioned, so everything stays float32 and the inverse comes from an
    # in-place Cholesky solve (peak ~3.5x the matrix size, vs ~9x for
    # np.linalg.inv, which gets OOM-killed on modest machines at this size).
    gram = (matrix.T @ matrix).toarray().astype(np.float32)
    diagonal = np.diag_indices(gram.shape[0])
    gram[diagonal] += l2
    popularity = np.asarray(matrix.sum(axis=0)).ravel()
    progress.update(1)
    factor = linalg.cho_factor(gram, overwrite_a=True, check_finite=False)
    precision = linalg.cho_solve(
        factor,
        np.eye(n_items, dtype=np.float32),
        overwrite_b=True,
        check_finite=False,
    )
    del factor, gram
    progress.update(2)
    diag_values = np.diag(precision).copy()
    precision /= -diag_values
    precision[diagonal] = 0.0
    progress.update(3)
    progress.finish()
    save_artifact(
        output_path,
        model_type="ease",
        item_ids=item_ids,
        weights=precision,
        popularity=popularity,
        l2=np.array([l2], dtype=np.float32),
    )


def train_sequential(interactions: pd.DataFrame, item_ids: list, output_path: Path) -> None:
    item_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
    transitions = np.zeros((len(item_ids), len(item_ids)), dtype=np.float32)
    grouped = interactions.groupby("user_id")
    progress = ProgressBar(grouped.ngroups, label="[3/3] Sequential CF transitions")
    for position, (_, group) in enumerate(grouped, start=1):
        if "timestamp" in group.columns:
            group = group.sort_values("timestamp")
        sequence = [item for item in group["item_id"].tolist() if item in item_index]
        for current, nxt in zip(sequence, sequence[1:]):
            transitions[item_index[current], item_index[nxt]] += 1.0
        progress.update(position)
    progress.finish()

    row_sums = transitions.sum(axis=1, keepdims=True)
    transitions = np.divide(
        transitions,
        row_sums,
        out=np.zeros_like(transitions),
        where=row_sums > 0,
    )
    popularity = interactions.groupby("item_id").size().reindex(item_ids, fill_value=0).to_numpy()
    save_artifact(
        output_path,
        model_type="sequential_cf",
        item_ids=item_ids,
        weights=transitions,
        popularity=popularity,
    )


def keep_topk(matrix: np.ndarray, k: int, progress: ProgressBar | None = None) -> None:
    if k <= 0 or k >= matrix.shape[1]:
        if progress is not None:
            progress.update(matrix.shape[0])
        return
    for row in range(matrix.shape[0]):
        keep = np.argpartition(matrix[row], -k)[-k:]
        mask = np.ones(matrix.shape[1], dtype=bool)
        mask[keep] = False
        matrix[row, mask] = 0.0
        if progress is not None:
            progress.update(row + 1)


def save_artifact(
    path: Path,
    *,
    model_type: str,
    item_ids: list,
    weights: np.ndarray,
    popularity: np.ndarray,
    **extra,
) -> None:
    np.savez_compressed(
        path,
        model_type=np.array([model_type]),
        item_ids=np.asarray(item_ids),
        weights=weights,
        popularity=popularity.astype(np.float32),
        **extra,
    )


def require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


if __name__ == "__main__":
    main()
