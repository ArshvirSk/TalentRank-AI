"""
Stage 7 — Final Ranking Engine (CLI entrypoint).

Owner: Member 3

Loads precomputed artifacts (embeddings, FAISS index), applies Stage 5
composite scoring + Stage 6 honeypot exclusion, tie-breaks by
candidate_id ascending, and writes the top-100 CSV.

Usage:
    python -m src.ranking.rank \\
        --candidates ./data/candidates.jsonl \\
        --jd ./data/job_description.docx \\
        --out ./submission.csv

Compute constraints (this script must satisfy all of these):
    - Wall-clock: ≤ 5 minutes
    - RAM:        ≤ 16 GB
    - Hardware:   CPU-only (no GPU)
    - Network:    none (fully offline)
    - LLMs:      no hosted LLM API calls
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import config


def run_ranking_pipeline(
    candidates_path: str | Path,
    jd_path: str | Path,
    output_path: str | Path,
) -> None:
    """
    Execute the full ranking pipeline and write the submission CSV.

    This function orchestrates:
      1. Load precomputed artifacts (embeddings, FAISS index).
      2. Parse JD into structured profile (Stage 1 — fast, cached).
      3. For each candidate in the FAISS top-K retrieval set:
         a. Extract career signals (Stage 2).
         b. Detect honeypots (Stage 6) — exclude flagged candidates.
         c. Compute composite score (Stage 5).
      4. Sort by final_score descending, tie-break by candidate_id ascending.
      5. Take top 100.
      6. Generate reasoning strings (Stage 8).
      7. Write CSV: candidate_id, rank, score, reasoning.

    Parameters
    ----------
    candidates_path : str or Path
        Path to candidates.jsonl.
    jd_path : str or Path
        Path to job_description.docx.
    output_path : str or Path
        Path to write the submission CSV.

    Raises
    ------
    NotImplementedError
        Scaffold — not yet implemented.
    RuntimeError
        If wall-clock time exceeds config.MAX_RANKING_WALL_CLOCK_SECONDS.
    """
    start_time = time.monotonic()

    # TODO(Member 3): implement the full ranking pipeline orchestration
    #
    # Pseudocode:
    #   jd_profile = retrieval.parse_job_description(jd_path)
    #   candidate_embeddings = np.load(config.CANDIDATE_EMBEDDINGS_PATH)
    #   jd_embeddings = np.load(config.JD_EMBEDDINGS_PATH)
    #
    #   # FAISS retrieval — get top-K candidates per JD vector
    #   distances, indices = retrieval.query_faiss_index(jd_embeddings)
    #
    #   # Merge retrieval sets, deduplicate
    #   candidate_pool = merge_retrieval_results(distances, indices)
    #
    #   scored = []
    #   for cid in candidate_pool:
    #       record = load_candidate(candidates_path, cid)
    #       signals = features.extract_career_signals(record)
    #       honeypot = features.detect_honeypot(record)
    #       if honeypot.is_honeypot:
    #           continue  # hard gate
    #       avail_mult = ranking.compute_availability_multiplier(record)
    #       breakdown = ranking.compute_composite_score(...)
    #       scored.append(breakdown)
    #
    #   # Sort: score desc, then candidate_id asc for tie-breaking
    #   scored.sort(key=lambda s: (-s.final_score, s.candidate_id))
    #   top_100 = scored[:config.TOP_K]
    #
    #   # Optional reranker (stretch goal)
    #   if config.RERANKER_ENABLED:
    #       top_100 = apply_reranker(top_100)
    #
    #   # Generate reasoning
    #   rows = []
    #   for rank, breakdown in enumerate(top_100, start=1):
    #       reasoning = explain.generate_reasoning(breakdown, jd_profile, rank)
    #       rows.append((breakdown.candidate_id, rank, breakdown.final_score, reasoning))
    #
    #   # Write CSV
    #   write_submission_csv(output_path, rows)

    elapsed = time.monotonic() - start_time
    if elapsed > config.MAX_RANKING_WALL_CLOCK_SECONDS:
        raise RuntimeError(
            f"Ranking exceeded time limit: {elapsed:.1f}s > "
            f"{config.MAX_RANKING_WALL_CLOCK_SECONDS}s"
        )

    raise NotImplementedError(
        "run_ranking_pipeline: pipeline orchestration not yet implemented"
    )


def write_submission_csv(
    output_path: str | Path,
    rows: list[tuple[str, int, float, str]],
) -> None:
    """
    Write the final submission CSV.

    Parameters
    ----------
    output_path : str or Path
        Destination file path.
    rows : list[tuple]
        List of (candidate_id, rank, score, reasoning) tuples.
    """
    # TODO(Member 3): implement CSV writer
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for row in rows:
            writer.writerow(row)

    print(f"Submission written to {output_path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    """Parse CLI arguments and run the ranking pipeline."""
    parser = argparse.ArgumentParser(
        description="TalentRank AI — Final Ranking Engine",
        epilog=(
            "Constraints: ≤5 min wall-clock, ≤16 GB RAM, CPU-only, "
            "no network, no hosted LLM calls."
        ),
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
        "--out", type=str, default=str(config.SUBMISSION_PATH),
        help="Output CSV path (default: submission.csv)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("TalentRank AI — Ranking Pipeline")
    print(f"  Candidates: {args.candidates}")
    print(f"  JD:         {args.jd}")
    print(f"  Output:     {args.out}")
    print("=" * 60)

    run_ranking_pipeline(
        candidates_path=args.candidates,
        jd_path=args.jd,
        output_path=args.out,
    )


if __name__ == "__main__":
    main()
