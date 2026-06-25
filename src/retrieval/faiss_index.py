"""
Stage 4 — FAISS Semantic Retrieval.

Owner: Member 2

Builds and queries a FAISS CPU index over candidate embeddings.  The index
is queried against three JD requirement vectors separately:

  - must-have similarity   → high = good
  - nice-to-have similarity → moderate bonus
  - disqualifier similarity → high = BAD (catches keyword stuffers)

This separation is what allows the pipeline to distinguish plain-language
good-fits from keyword-stuffer bad-fits.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

import config


def build_faiss_index(
    embeddings_path: str | Path | None = None,
    index_output_path: str | Path | None = None,
) -> None:
    """
    Build a FAISS CPU index from precomputed candidate embeddings.

    Parameters
    ----------
    embeddings_path : str or Path, optional
        Path to the .npy file with candidate embeddings.
        Defaults to config.CANDIDATE_EMBEDDINGS_PATH.
    index_output_path : str or Path, optional
        Where to write the .faiss index file.
        Defaults to config.FAISS_INDEX_PATH.

    Raises
    ------
    NotImplementedError
        Scaffold — not yet implemented.

    Notes
    -----
    Implementation checklist (Member 2):
      1. Load embeddings from .npy (shape: [N, D]).
      2. Choose index type: IndexFlatIP for exact search (100k is small
         enough) or IndexIVFFlat for speed if profiling shows a need.
      3. Train (if IVF) and add vectors.
      4. Write index to disk.
    """
    # TODO(Member 2): implement FAISS index build
    raise NotImplementedError(
        "build_faiss_index: index construction not yet implemented"
    )


def query_faiss_index(
    query_vectors: NDArray[np.float32],
    top_k: int = config.FAISS_TOP_K_RETRIEVAL,
    index_path: str | Path | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """
    Query the FAISS index with one or more vectors.

    Parameters
    ----------
    query_vectors : NDArray[np.float32]
        Query matrix of shape ``(num_queries, D)``.  Typically 3 vectors:
        [must_have_embedding, nice_to_have_embedding, disqualifier_embedding].
    top_k : int
        Number of nearest neighbours to retrieve per query.
    index_path : str or Path, optional
        Path to the .faiss index file.
        Defaults to config.FAISS_INDEX_PATH.

    Returns
    -------
    distances : NDArray[np.float32]
        Shape ``(num_queries, top_k)`` — similarity scores.
    indices : NDArray[np.int64]
        Shape ``(num_queries, top_k)`` — candidate row indices.

    Raises
    ------
    NotImplementedError
        Scaffold — not yet implemented.

    Notes
    -----
    Implementation checklist (Member 2):
      1. Load FAISS index from disk.
      2. Set nprobe if IVF index (config.FAISS_NPROBE).
      3. Search and return distances + indices.
      4. Consider normalizing query vectors if using inner-product metric.
    """
    # TODO(Member 2): implement FAISS querying
    raise NotImplementedError(
        "query_faiss_index: index querying not yet implemented"
    )
