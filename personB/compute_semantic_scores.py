"""
person_b/compute_semantic_scores.py

Run AFTER precompute_embeddings.py.
Loads precomputed matrix and produces semantic_scores.parquet for Person C.

Produces:
  semantic_scores.parquet with columns:
    candidate_id, sim_must, sim_nice, sim_disq, raw_semantic_score

Runtime: < 5 seconds for 100k candidates (pure matrix multiply)

Usage:
  python compute_semantic_scores.py \
    --precomputed_dir ./precomputed \
    --out semantic_scores.parquet

  # Sample mode (uses ./precomputed_sample, writes semantic_scores_sample.parquet)
  python compute_semantic_scores.py --sample
"""

import argparse
import time
import numpy as np
import pandas as pd

# Scoring weight constants — used for raw_semantic_score convenience column
W_MUST = 0.6
W_NICE = 0.2
W_DISQ = 0.4

# Top-N for sanity report
TOP_N = 5


def validate_scores(scores: pd.DataFrame, n_expected: int) -> None:
    """Validate score distributions before saving. Raises RuntimeError on any failure.

    Does NOT save the file if validation fails.
    Requirements: 7.3, 7.4, 11.1, 11.2, 11.3, 11.4
    """
    # Row count
    if len(scores) != n_expected:
        raise RuntimeError(
            f"Validation failed — row count: observed={len(scores)}, expected={n_expected}"
        )
    # sim_must max < 0.95
    must_max = float(scores["sim_must"].max())
    if must_max >= 0.95:
        raise RuntimeError(
            f"Validation failed — sim_must.max(): observed={must_max:.4f}, expected<0.95"
        )
    # mean(sim_disq) < mean(sim_must)
    mean_must = float(scores["sim_must"].mean())
    mean_disq = float(scores["sim_disq"].mean())
    if mean_disq >= mean_must:
        raise RuntimeError(
            f"Validation failed — mean(sim_disq)={mean_disq:.4f} >= mean(sim_must)={mean_must:.4f}"
        )
    # IQR(raw_semantic_score) > 0.01
    q75 = float(np.percentile(scores["raw_semantic_score"], 75))
    q25 = float(np.percentile(scores["raw_semantic_score"], 25))
    iqr = q75 - q25
    if iqr <= 0.01:
        raise RuntimeError(
            f"Validation failed — IQR(raw_semantic_score)={iqr:.4f}, expected>0.01"
        )


def main(precomputed_dir: str, out_path: str):
    t0 = time.time()

    # Load precomputed artifacts
    matrix = np.load(f"{precomputed_dir}/candidate_embeddings.npy")   # (N, 384)
    index  = pd.read_parquet(f"{precomputed_dir}/candidate_index.parquet")
    qv     = np.load(f"{precomputed_dir}/jd_query_vectors.npy")        # (3, 384)

    print(f"Matrix: {matrix.shape} | Query vectors: {qv.shape}")

    # Cosine similarity = dot product (both are L2-normalized)
    # Shape: (N,) for each
    sim_must = matrix @ qv[0]   # must-have similarity
    sim_nice = matrix @ qv[1]   # nice-to-have similarity
    sim_disq = matrix @ qv[2]   # disqualifier similarity

    # Composite convenience score
    raw_semantic_score = (W_MUST * sim_must + W_NICE * sim_nice - W_DISQ * sim_disq).astype("float32")

    scores = pd.DataFrame({
        "candidate_id":       index["candidate_id"].values,
        "sim_must":           sim_must.astype("float32"),
        "sim_nice":           sim_nice.astype("float32"),
        "sim_disq":           sim_disq.astype("float32"),
        "raw_semantic_score": raw_semantic_score,
    })

    # Validate scores before saving
    validate_scores(scores, len(index))

    # Print one-line distribution summary
    for col in ["sim_must", "sim_nice", "sim_disq"]:
        s = scores[col]
        print(f"  {col}: min={s.min():.4f} mean={s.mean():.4f} max={s.max():.4f} std={s.std():.4f}")

    scores.to_parquet(out_path, index=False)
    print(f"Saved {out_path} — {len(scores):,} rows in {time.time()-t0:.2f}s")

    # Sanity report — top-N by sim_must and raw_semantic_score
    print(f"\nTop {TOP_N} by sim_must (sanity check):")
    print(scores.nlargest(TOP_N, "sim_must").to_string(index=False))
    print(f"\nTop {TOP_N} by raw_semantic_score:")
    print(scores.nlargest(TOP_N, "raw_semantic_score").to_string(index=False))

    # Distribution stats
    print("\nScore distributions:")
    print(scores[["sim_must", "sim_nice", "sim_disq", "raw_semantic_score"]].describe().round(4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--precomputed_dir", default="./precomputed")
    parser.add_argument("--out", default="./semantic_scores.parquet")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use precomputed_sample dir and output semantic_scores_sample.parquet",
    )
    args = parser.parse_args()

    precomputed_dir = args.precomputed_dir
    out_path = args.out
    if args.sample:
        if args.precomputed_dir == "./precomputed":
            precomputed_dir = "./precomputed_sample"
        if args.out == "./semantic_scores.parquet":
            out_path = "./semantic_scores_sample.parquet"
    main(precomputed_dir, out_path)
