"""
person_b/precompute_embeddings.py

Run ONCE offline before the hackathon submission.
Produces:
  - candidate_embeddings.npy   : float32 (N, 384) matrix, L2-normalized
  - candidate_index.parquet    : [candidate_id, idx] mapping
  - jd_query_vectors.npy       : float32 (3, 384) — [must, nice, disq]

Runtime: ~15-25 min for 100k on 4-core CPU (BGE-small)
Memory:  ~1.5 GB peak

Usage:
  python precompute_embeddings.py \
    --input candidates.jsonl \
    --out_dir ./precomputed
"""

import argparse
import glob
import gzip
import json
import logging
import os
import sys
import time
import warnings
from multiprocessing import Pool, cpu_count

# ── Windows DLL fix ─────────────────────────────────────────────────────────
# PyTorch c10.dll fails to load in child processes on Windows unless the
# torch lib directory is explicitly on PATH before any import.
if sys.platform == "win32":
    try:
        import torch
        _torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if _torch_lib not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass  # torch not installed — will fail later with a clear message

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from build_texts import (
    build_candidate_text,
    JD_MUST_HAVE,
    JD_NICE_TO_HAVE,
    JD_DISQUALIFIER,
)

MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64

BGE_QUERY_PREFIX = "Represent this sentence: "

logger = logging.getLogger(__name__)


def _is_bge_model(model_name: str) -> bool:
    """Return True if model_name refers to a BGE-family model (case-insensitive)."""
    return "bge" in model_name.lower()


def _prepare_query_texts(texts: list[str], model_name: str) -> list[str]:
    """Prepend BGE asymmetric retrieval prefix to query texts when using a BGE model.

    For non-BGE models, returns texts unchanged.
    The prefix is applied only to JD query texts, never to candidate document texts.

    Requirements: 4.1, 4.4
    """
    if _is_bge_model(model_name):
        return [BGE_QUERY_PREFIX + t for t in texts]
    return texts


# ── Checkpoint helpers ───────────────────────────────────────────────────────

def _save_checkpoint(out_dir: str, chunk_idx: int, ids: list, embeddings: np.ndarray) -> None:
    """Atomically save a checkpoint shard to <out_dir>/checkpoints/chunk_<i>.ckpt.npz.

    Uses write-to-temp + os.rename for atomicity (no corrupt partial writes).
    Requirements: 5.1, 5.4
    """
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    final_path = os.path.join(ckpt_dir, f"chunk_{chunk_idx}.ckpt.npz")
    # np.savez automatically appends .npz — use a tmp name without the extension
    tmp_base   = os.path.join(ckpt_dir, f"chunk_{chunk_idx}.ckpt.tmp")
    np.savez(tmp_base,
             candidate_ids=np.array(ids, dtype=object),
             embeddings=embeddings.astype("float32"))
    # np.savez wrote to tmp_base + ".npz"
    os.replace(tmp_base + ".npz", final_path)  # atomic on POSIX and Windows


def _load_checkpoints(out_dir: str) -> tuple[set, list]:
    """Scan <out_dir>/checkpoints/ for .ckpt.npz files and load them.

    Returns:
        done_ids: set of candidate_id strings already processed
        shards:   list of (chunk_idx, candidate_ids_array, embeddings_array) tuples

    Corrupt checkpoint files log a WARNING and are skipped (deleted).
    Requirements: 5.2, 5.5
    """
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    done_ids: set = set()
    shards: list = []

    if not os.path.isdir(ckpt_dir):
        return done_ids, shards

    pattern = os.path.join(ckpt_dir, "chunk_*.ckpt.npz")
    for path in sorted(glob.glob(pattern)):
        try:
            with np.load(path, allow_pickle=True) as data:
                ids_arr = data["candidate_ids"].copy()
                emb_arr = data["embeddings"].copy()
            done_ids.update(str(cid) for cid in ids_arr)
            # Extract chunk index from filename
            chunk_idx = int(os.path.basename(path).split("_")[1].split(".")[0])
            shards.append((chunk_idx, ids_arr, emb_arr))
        except Exception as e:
            logger.warning(f"Corrupt checkpoint {path}: {e} — deleting and reprocessing")
            try:
                os.remove(path)
            except OSError:
                pass

    return done_ids, shards


def _merge_checkpoints(out_dir: str, all_ids: list, dim: int = 384) -> np.ndarray:
    """Merge all checkpoint shards into a correctly-ordered embedding matrix.

    Reconstructs original candidate order using all_ids as the source of truth.
    Deletes the checkpoint directory after successful merge.

    Requirements: 5.3
    """
    import shutil

    _done_ids, shards = _load_checkpoints(out_dir)

    # Build id → embedding row lookup
    id_to_emb: dict = {}
    for _chunk_idx, ids_arr, emb_arr in shards:
        for cid, emb in zip(ids_arr, emb_arr):
            id_to_emb[str(cid)] = emb

    # Reconstruct in original order
    matrix = np.zeros((len(all_ids), dim), dtype="float32")
    for i, cid in enumerate(all_ids):
        if cid in id_to_emb:
            matrix[i] = id_to_emb[cid]

    # Delete checkpoint directory
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    if os.path.isdir(ckpt_dir):
        try:
            shutil.rmtree(ckpt_dir)
            print(f"  Checkpoints merged and cleaned up")
        except (PermissionError, OSError):
            # Windows may hold file handles briefly; leave the dir, it won't affect correctness
            print(f"  Checkpoints merged (cleanup skipped — rerun will resume from cache)")

    return matrix


def _default_workers() -> int:
    """Compute RAM-aware default worker count.
    
    Formula: max(1, min(cpu_count(), floor(free_ram_gb / 2.5), 8))
    Falls back to min(cpu_count(), 4) if psutil is unavailable.
    
    Requirements: 9.1, 9.4
    """
    try:
        import psutil
        free_gb = psutil.virtual_memory().available / 1e9
        return max(1, min(cpu_count(), int(free_gb / 2.5), 8))
    except Exception:
        warnings.warn(
            "psutil not available; defaulting to min(cpu_count(), 4). "
            "Install with: pip install psutil",
            stacklevel=2,
        )
        return min(cpu_count(), 4)


# ── Worker function (each process loads its own model copy) ──────────────────

def validate_embeddings(matrix: np.ndarray, name: str) -> None:
    """Validate an embedding matrix before saving.

    Checks:
    - Second dimension is 384
    - No NaN values
    - All row L2-norms within [0.999, 1.001]

    Raises RuntimeError with observed value on any failure.
    Requirements: 7.1, 7.2
    """
    if matrix.shape[1] != 384:
        raise RuntimeError(
            f"Validation failed — {name}: shape[-1]={matrix.shape[1]}, expected=384"
        )
    if np.isnan(matrix).any():
        n_nans = int(np.isnan(matrix).sum())
        raise RuntimeError(
            f"Validation failed — {name}: contains {n_nans} NaN value(s)"
        )
    norms = np.linalg.norm(matrix, axis=1)
    bad_mask = (norms < 0.999) | (norms > 1.001)
    if bad_mask.any():
        n_bad = int(bad_mask.sum())
        raise RuntimeError(
            f"Validation failed — {name}: {n_bad} row(s) have L2-norm outside [0.999, 1.001]"
        )


def embed_chunk(args: tuple) -> np.ndarray:
    # Ensure torch DLLs are findable in Windows child processes
    import sys, os
    if sys.platform == "win32":
        try:
            import torch as _t
            _lib = os.path.join(os.path.dirname(_t.__file__), "lib")
            if _lib not in os.environ.get("PATH", ""):
                os.environ["PATH"] = _lib + os.pathsep + os.environ.get("PATH", "")
        except Exception:
            pass
    texts, model_name = args
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,   # L2 norm → cosine sim = dot product
        show_progress_bar=False,
    ).astype("float32")


# ── Main ─────────────────────────────────────────────────────────────────────

def main(input_path: str, out_dir: str, workers: int | None = None, sample: bool = False):
    from tqdm import tqdm

    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    # 1. Load candidates
    print(f"Loading candidates from {input_path} ...")
    candidates = []
    opener = gzip.open if input_path.endswith(".gz") else open
    with opener(input_path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"  Loaded {len(candidates):,} candidates in {time.time()-t0:.1f}s")

    # Capture full ID list before resume filtering (needed for final merge)
    all_original_ids = [c["candidate_id"] for c in candidates]

    # Check for existing checkpoints (resume support)
    done_ids, _existing_shards = _load_checkpoints(out_dir)
    if done_ids:
        before = len(candidates)
        candidates = [c for c in candidates if c["candidate_id"] not in done_ids]
        print(f"  Resuming: skipping {before - len(candidates):,} already-checkpointed candidates, "
              f"{len(candidates):,} remaining")

    # Sample mode: limit to first 50 candidates
    if sample:
        candidates = candidates[:50]
        print(f"  [--sample] Limited to {len(candidates)} candidates")

    # Update all_original_ids to reflect the actual set being processed
    # (after resume filtering AND sample slicing)
    all_original_ids = [c["candidate_id"] for c in candidates]

    # 2. Build text representations
    print("Building candidate texts ...")
    ids   = [c["candidate_id"] for c in candidates]
    texts = [build_candidate_text(c) for c in candidates]
    print(f"  Done. Median word count: {int(np.median([len(t.split()) for t in texts]))}")

    # 3. Embed in parallel across CPU cores
    if workers is not None:
        if workers < 1:
            raise ValueError("--workers must be >= 1")
        n_workers = workers
    else:
        n_workers = _default_workers()
    try:
        import psutil as _psutil
        free_gb = _psutil.virtual_memory().available / 1e9
        print(f"Embedding with {MODEL_NAME} on {n_workers} workers | Available RAM: {free_gb:.1f} GB ...")
    except Exception:
        print(f"Embedding with {MODEL_NAME} on {n_workers} workers ...")
    chunks = [texts[i::n_workers] for i in range(n_workers)]
    chunk_args = [(chunk, MODEL_NAME) for chunk in chunks]

    t1 = time.time()
    results = []
    with Pool(n_workers) as pool:
        with tqdm(total=len(candidates), unit="cand", desc="Embedding", disable=sample) as pbar:
            for chunk_result in pool.imap(embed_chunk, chunk_args):
                results.append(chunk_result)
                pbar.update(len(chunk_result))

    # Reconstruct current-batch matrix
    dim    = results[0].shape[1]
    matrix = np.zeros((len(ids), dim), dtype="float32")
    for i, chunk_result in enumerate(results):
        matrix[i::n_workers] = chunk_result

    print(f"  Embedding done in {time.time()-t1:.1f}s | matrix shape: {matrix.shape}")

    # Save per-chunk checkpoints after embedding
    chunk_ids_list = [ids[i::n_workers] for i in range(n_workers)]
    for i, (chunk_result, cids) in enumerate(zip(results, chunk_ids_list)):
        _save_checkpoint(out_dir, i, cids, chunk_result)
    print(f"  Checkpoints saved ({n_workers} shards)")

    # Merge all shards (including any from previous runs) into final ordered matrix
    matrix = _merge_checkpoints(out_dir, all_original_ids, dim)
    print(f"  Final matrix shape after merge: {matrix.shape}")

    # Validate candidate embeddings
    validate_embeddings(matrix, "candidate_embeddings")
    print("  Validation passed: candidate_embeddings")

    # Use all_original_ids as the authoritative ID list for the index file
    ids = all_original_ids

    # 4. Save candidate matrix + ID index
    emb_path = os.path.join(out_dir, "candidate_embeddings.npy")
    idx_path = os.path.join(out_dir, "candidate_index.parquet")
    np.save(emb_path, matrix)
    pd.DataFrame({"candidate_id": ids, "idx": range(len(ids))}).to_parquet(idx_path, index=False)
    print(f"  Saved {emb_path} ({os.path.getsize(emb_path)/1e6:.1f} MB)")
    print(f"  Saved {idx_path}")

    # 5. Embed the 3 JD query vectors (fast — only 3 texts)
    print("Embedding JD query vectors ...")
    model = SentenceTransformer(MODEL_NAME)
    query_texts = _prepare_query_texts(
        [JD_MUST_HAVE, JD_NICE_TO_HAVE, JD_DISQUALIFIER],
        MODEL_NAME,
    )
    query_matrix = model.encode(
        query_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    qv_path = os.path.join(out_dir, "jd_query_vectors.npy")
    np.save(qv_path, query_matrix)
    # Validate JD query vectors
    validate_embeddings(query_matrix, "jd_query_vectors")
    print(f"  Saved {qv_path} — shape {query_matrix.shape} (validation passed)")
    print(f"\nTotal precompute time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   default="candidates.jsonl", help="Path to candidates.jsonl or .jsonl.gz")
    parser.add_argument("--out_dir", default="./precomputed",    help="Directory to save outputs")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Process only the first 50 candidates (for fast pipeline testing)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel embedding workers (default: RAM-aware auto)",
    )
    args = parser.parse_args()
    # --sample implies ./precomputed_sample unless --out_dir was explicitly set
    out_dir = args.out_dir
    if args.sample and args.out_dir == "./precomputed":
        out_dir = "./precomputed_sample"
    main(args.input, out_dir, workers=args.workers, sample=args.sample)
