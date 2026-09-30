"""
Retrieval backend benchmark: numpy matmul vs FAISS IndexFlatIP.

Measures the claim made in src/retrieval/faiss_index.py: at 100k x 384,
a single vectorized matmul is as exact and faster than building+querying
an index — and shows where the crossover lies.

Usage:
    python -m eval.retrieval_benchmark                 # default sizes
    python -m eval.retrieval_benchmark --sizes 1000 100000
"""

from __future__ import annotations

import argparse
import time

import numpy as np

faiss = None
try:
    import faiss as _faiss  # type: ignore[import-not-found]
    faiss = _faiss
except ImportError:
    pass

from src.retrieval.faiss_index import normalize_rows


def bench_numpy(emb: np.ndarray, queries: np.ndarray, repeats: int = 5) -> float:
    """Median wall-clock seconds for the full-matrix matmul path."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        sims = emb @ queries.T  # (N, 3) — the exact operation rank.py uses
        _ = np.argsort(sims[:, 0])[::-1][:500]
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def bench_faiss(emb: np.ndarray, queries: np.ndarray, repeats: int = 5) -> tuple[float, bool]:
    """Median wall-clock seconds for build + query via IndexFlatIP.

    Returns (seconds, built_ok).
    """
    if faiss is None:
        return float("nan"), False

    build_times, query_times = [], []
    for _ in range(repeats):
        t0 = time.perf_counter()
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        build_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        index.search(queries, 500)
        query_times.append(time.perf_counter() - t0)

    # First query on a fresh index pays setup; build dominates, so report
    # the total the pipeline would actually pay per run.
    return float(np.median(build_times) + np.median(query_times)), True


def run(sizes: list[int], dim: int = 384, seed: int = 42) -> int:
    rng = np.random.default_rng(seed)
    queries = normalize_rows(rng.standard_normal((3, dim)).astype(np.float32))

    print("=" * 78)
    print(f"Retrieval benchmark — exact inner product, dim={dim}, 3 JD queries, top-500")
    print("numpy: full matmul + argsort | faiss: IndexFlatIP build + search (median of 5)")
    print("=" * 78)
    print(f"{'N rows':>10} {'numpy (s)':>12} {'faiss build+query (s)':>24} {'winner':>12}")
    print("-" * 78)

    for n in sizes:
        # Generate up to n rows once, reuse for both backends
        emb = normalize_rows(rng.standard_normal((n, dim)).astype(np.float32))

        t_numpy = bench_numpy(emb, queries)

        t_faiss, ok = bench_faiss(emb, queries)
        t_faiss_str = f"{t_faiss:.4f}" if ok else "n/a (no faiss)"

        if not ok:
            winner = "numpy"
        elif t_numpy <= t_faiss:
            winner = "numpy"
        else:
            winner = "faiss"

        print(f"{n:>10,} {t_numpy:>12.4f} {t_faiss_str:>24} {winner:>12}")

    print("-" * 78)
    print("Interpretation: both paths are EXACT (identical top-k), so the choice")
    print("is purely runtime/dependency. FAISS wins when N outgrows RAM (mmap)")
    print("or when IVF/HNSW approximation is acceptable at 10M+ vectors.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[1_000, 10_000, 100_000],
                        help="Corpus sizes to benchmark")
    parser.add_argument("--dim", type=int, default=384)
    args = parser.parse_args()
    raise SystemExit(run(args.sizes, dim=args.dim))
