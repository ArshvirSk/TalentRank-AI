"""
Stage 3 — Candidate Embedding Generation.

Owner: Member 2

CPU-friendly local sentence-transformers model (e.g. BAAI/bge-small-en-v1.5
or intfloat/e5-small-v2).  Embeddings are precomputed once offline for all
100k candidate profiles plus the JD's requirement vectors.

This module is also runnable as a script for batch precomputation:
    python -m src.retrieval.embed \\
        --candidates ./data/candidates.jsonl \\
        --jd ./data/job_description.docx
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

import config


def _load_model():
    """
    Load the sentence-transformer model specified in config.

    Returns
    -------
    SentenceTransformer
        Loaded model ready for encoding.

    Raises
    ------
    NotImplementedError
        Scaffold — not yet implemented.
    """
    # TODO(Member 2): load model from config.EMBEDDING_MODEL_NAME
    # Ensure device='cpu' is explicit.
    raise NotImplementedError("_load_model: model loading not yet implemented")


def embed_texts(
    texts: list[str],
    batch_size: int = 256,
    show_progress: bool = True,
) -> NDArray[np.float32]:
    """
    Embed a list of texts using the configured sentence-transformer.

    Parameters
    ----------
    texts : list[str]
        Input texts to embed.
    batch_size : int
        Encoding batch size (tune for available RAM).
    show_progress : bool
        Whether to display a tqdm progress bar.

    Returns
    -------
    NDArray[np.float32]
        Embedding matrix of shape ``(len(texts), config.EMBEDDING_DIMENSION)``.

    Raises
    ------
    NotImplementedError
        Scaffold — not yet implemented.
    """
    # TODO(Member 2): implement batch embedding with tqdm progress
    raise NotImplementedError("embed_texts: embedding not yet implemented")


def precompute_all_embeddings(
    candidates_path: str | Path,
    jd_path: str | Path,
    output_dir: str | Path | None = None,
) -> None:
    """
    Batch-precompute embeddings for all candidates and the JD vectors,
    saving results as .npy files.

    This is the main offline precomputation step (no time limit).

    Parameters
    ----------
    candidates_path : str or Path
        Path to candidates.jsonl.
    jd_path : str or Path
        Path to job_description.docx.
    output_dir : str or Path, optional
        Directory to write .npy files.  Defaults to config.ARTIFACTS_DIR.

    Raises
    ------
    NotImplementedError
        Scaffold — not yet implemented.

    Notes
    -----
    Implementation checklist (Member 2):
      1. Parse JD via jd_parser.parse_job_description → embed must-have,
         nice-to-have, disqualifier texts → save as jd_embeddings.npy.
      2. Stream candidates.jsonl, extract a text representation per
         candidate (concatenation of summary + skills + career descriptions),
         embed in batches → save as candidate_embeddings.npy.
      3. Save a mapping file (candidate_id → row index) for the embeddings.
    """
    # TODO(Member 2): implement full offline precompute pipeline
    raise NotImplementedError(
        "precompute_all_embeddings: precompute pipeline not yet implemented"
    )


# ---------------------------------------------------------------------------
# CLI entrypoint for offline precomputation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Precompute embeddings for candidates and JD"
    )
    parser.add_argument(
        "--candidates", type=str, required=True,
        help="Path to candidates.jsonl",
    )
    parser.add_argument(
        "--jd", type=str, required=True,
        help="Path to job_description.docx",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save .npy artifacts (default: config.ARTIFACTS_DIR)",
    )
    args = parser.parse_args()

    precompute_all_embeddings(
        candidates_path=args.candidates,
        jd_path=args.jd,
        output_dir=args.output_dir,
    )
