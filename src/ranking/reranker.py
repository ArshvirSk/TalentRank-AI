"""
Stage 7 (optional stretch) — LightGBM/XGBoost Reranker.

Owner: Member 3

An optional reranker trained on rule-based pseudo-labels (since there is
no ground truth) for score calibration.  This module is **not** a
dependency of the baseline ranking path — rank.py works without it.

If ``config.RERANKER_ENABLED`` is False, the reranker is skipped entirely.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

import config


def train_reranker(
    features: NDArray[np.float32],
    pseudo_labels: NDArray[np.float32],
    output_path: str | Path | None = None,
) -> None:
    """
    Train a LightGBM reranker on pseudo-labelled feature vectors.

    Parameters
    ----------
    features : NDArray[np.float32]
        Feature matrix of shape ``(N, num_features)``.
    pseudo_labels : NDArray[np.float32]
        Pseudo-label scores of shape ``(N,)`` derived from rule-based
        ranking (Stage 5 output).
    output_path : str or Path, optional
        Where to save the trained model.
        Defaults to config.RERANKER_MODEL_PATH.

    Raises
    ------
    NotImplementedError
        Scaffold — not yet implemented.

    Notes
    -----
    Implementation checklist (Member 3):
      1. Create LightGBM Dataset from features and pseudo-labels.
      2. Train with LambdaRank objective (lambdarank or rank_xendcg).
      3. Validate with a held-out fold.
      4. Save model to disk.
    """
    # TODO(Member 3): implement reranker training (stretch goal)
    raise NotImplementedError(
        "train_reranker: stretch goal — not yet implemented"
    )


def apply_reranker(
    features: NDArray[np.float32],
    model_path: str | Path | None = None,
) -> NDArray[np.float32]:
    """
    Apply a pre-trained reranker to produce calibrated scores.

    Parameters
    ----------
    features : NDArray[np.float32]
        Feature matrix of shape ``(N, num_features)``.
    model_path : str or Path, optional
        Path to the saved LightGBM model.
        Defaults to config.RERANKER_MODEL_PATH.

    Returns
    -------
    NDArray[np.float32]
        Calibrated scores of shape ``(N,)``.

    Raises
    ------
    NotImplementedError
        Scaffold — not yet implemented.
    """
    # TODO(Member 3): implement reranker inference (stretch goal)
    raise NotImplementedError(
        "apply_reranker: stretch goal — not yet implemented"
    )
