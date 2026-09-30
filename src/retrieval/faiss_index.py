"""
Stage 4 — Semantic Retrieval (FAISS backend).

Owner: Member 2 (implemented post-hackathon by Arshvir — see docs/OWNERSHIP.md)

Builds and queries a FAISS CPU index over candidate embeddings.  The index
is queried against three JD requirement vectors separately:

  - must-have similarity    → high = good
  - nice-to-have similarity → moderate bonus
  - disqualifier similarity → high = BAD (catches keyword stuffers)

This separation is what allows the pipeline to distinguish plain-language
good-fits from keyword-stuffer bad-fits.

BACKEND NOTE (why numpy is the default at runtime)
--------------------------------------------------
Both backends compute EXACT inner-product search over L2-normalized rows,
so results are identical.  For this corpus (100,000 x 384 float32 ≈ 153 MB)
a single vectorized matmul computes all similarities faster than building
and querying an index, with zero extra runtime dependency.  FAISS pays off
when the corpus outgrows RAM (memory-mapped indexes) or when IVF/HNSW
approximation is acceptable (10M+ vectors).

  - config.RETRIEVAL_BACKEND = "numpy"  → exact matmul (default; see
    eval/retrieval_benchmark.py for the measured comparison)
  - config.RETRIEVAL_BACKEND = "faiss"  → IndexFlatIP through this module

`build_faiss_index` / `query_faiss_index` are fully implemented and tested
(tests/test_retrieval.py) so the FAISS path is one config flip away.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import config

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _load_faiss():
    """Import faiss with a clear error message when unavailable."""
    try:
        import faiss  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            "faiss-cpu is required for the FAISS retrieval backend. "
            "Install it with: pip install faiss-cpu  (or set "
            "config.RETRIEVAL_BACKEND = 'numpy' for the exact matmul path)"
        ) from exc
    return faiss


def normalize_rows(matrix: NDArray[np.floating]) -> NDArray[np.float32]:
    """L2-normalize rows defensively; zero rows are left as-is (norm → 1)."""
    x = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (x / norms).astype(np.float32)


def build_faiss_index(
    embeddings_path: str | Path | None = None,
    index_output_path: str | Path | None = None,
    embeddings: NDArray[np.floating] | None = None,
) -> Path:
    """
    Build a FAISS CPU index (IndexFlatIP — exact inner product) from
    precomputed candidate embeddings and write it to disk.

    Parameters
    ----------
    embeddings_path : str or Path, optional
        Path to the .npy file with candidate embeddings.
        Defaults to config.CANDIDATE_EMBEDDINGS_PATH.  Ignored when
        ``embeddings`` is passed directly (used by tests).
    index_output_path : str or Path, optional
        Where to write the .faiss index file.
        Defaults to config.FAISS_INDEX_PATH.
    embeddings : NDArray, optional
        In-process embedding matrix (N, D).  When provided, no file is read.

    Returns
    -------
    Path
        The path the index was written to.

    Notes
    -----
    IndexFlatIP over L2-normalized rows ranks by cosine similarity exactly —
    the same ordering as the numpy matmul backend.  An IVF variant would
    trade exactness for speed; at 100k rows exactness is free, so it isn't
    used (see module docstring).
    """
    faiss = _load_faiss()

    if embeddings is None:
        emb_path = Path(embeddings_path or config.CANDIDATE_EMBEDDINGS_PATH)
        embeddings = np.load(emb_path, allow_pickle=False)

    matrix = normalize_rows(embeddings)
    dim = matrix.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(matrix)  # type: ignore[attr-defined]

    out_path = Path(index_output_path or config.FAISS_INDEX_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_path))  # type: ignore[attr-defined]
    return out_path


def query_faiss_index(
    query_vectors: NDArray[np.floating],
    top_k: int = config.FAISS_TOP_K_RETRIEVAL,
    index_path: str | Path | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """
    Query the FAISS index with one or more vectors.

    Parameters
    ----------
    query_vectors : NDArray
        Query matrix of shape ``(num_queries, D)``.  Typically 3 vectors:
        [must_have_embedding, nice_to_have_embedding, disqualifier_embedding].
        Rows are L2-normalized here so inner product = cosine similarity.
    top_k : int
        Number of nearest neighbours to retrieve per query.
    index_path : str or Path, optional
        Path to the .faiss index file.
        Defaults to config.FAISS_INDEX_PATH.

    Returns
    -------
    distances : NDArray[np.float32]
        Shape ``(num_queries, top_k)`` — cosine similarity scores (descending).
    indices : NDArray[np.int64]
        Shape ``(num_queries, top_k)`` — candidate row indices.
    """
    faiss = _load_faiss()

    idx_path = Path(index_path or config.FAISS_INDEX_PATH)
    if not idx_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {idx_path} — build it first with "
            f"build_faiss_index() (or set config.RETRIEVAL_BACKEND = 'numpy')"
        )

    index = faiss.read_index(str(idx_path))  # type: ignore[attr-defined]
    queries = normalize_rows(np.asarray(query_vectors, dtype=np.float32))

    k = min(int(top_k), index.ntotal)
    distances, indices = index.search(queries, k)  # type: ignore[attr-defined]
    return distances, indices
