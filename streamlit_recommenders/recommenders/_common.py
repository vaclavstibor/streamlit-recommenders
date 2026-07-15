from __future__ import annotations

import numpy as np


def rank_scores(
    scores: np.ndarray,
    item_ids: list,
    seen: set,
    k: int,
) -> list:
    """Return the top-``k`` unseen item ids by descending score.

    Args:
        scores: One score per item, aligned with ``item_ids``.
        item_ids: Ordered catalog of item ids.
        seen: Item ids to exclude (already-seen history/session items).
        k: Maximum number of ids to return.

    Returns:
        Up to ``k`` item ids, best score first, skipping anything in ``seen``.
    """
    ranked = np.argsort(scores)[::-1]
    result = []
    for idx in ranked:
        item_id = item_ids[int(idx)]
        if item_id in seen:
            continue
        result.append(item_id)
        if len(result) >= k:
            break
    return result
