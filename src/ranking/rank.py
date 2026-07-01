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
import json
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

import config
from src.features.schema import CandidateRecord
from src.features.career_signals import extract_career_signals, CareerSignals
from src.features.honeypot import compute_honeypot_score
from src.ranking.behavioral import compute_availability_multiplier, compute_behavioral_score
from src.ranking.fusion import compute_composite_score, ScoreBreakdown


def _load_candidates_dict(candidates_path: str | Path) -> Dict[str, CandidateRecord]:
    """
    Load candidates.jsonl into a dict for fast lookup by candidate_id.
    
    Parameters
    ----------
    candidates_path : str or Path
        Path to candidates.jsonl
    
    Returns
    -------
    dict
        Mapping candidate_id -> CandidateRecord
    """
    candidates = {}
    candidates_path = Path(candidates_path)
    
    with open(candidates_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            try:
                # Convert dict to CandidateRecord if needed
                if isinstance(data, dict):
                    record = CandidateRecord(
                        candidate_id=data.get('candidate_id'),
                        profile=data.get('profile'),
                        career_history=data.get('career_history', []),
                        education=data.get('education', []),
                        skills=data.get('skills', []),
                        redrob_signals=data.get('redrob_signals', {}),
                    )
                else:
                    record = data
                
                candidates[record.candidate_id] = record
            except Exception as e:
                print(f"Warning: Failed to load candidate {data.get('candidate_id')}: {e}")
                continue
    
    return candidates


def _load_faiss_artifacts(
    embeddings_path: str | Path,
    index_path: str | Path,
) -> tuple:
    """
    Load FAISS index and embeddings from disk.
    
    Parameters
    ----------
    embeddings_path : str or Path
        Path to embeddings.npy (precomputed candidate embeddings).
    index_path : str or Path
        Path to index.faiss (FAISS index).
    
    Returns
    -------
    tuple(embeddings, index)
        embeddings: (n_candidates, embedding_dim) array
        index: FAISS index object
    """
    import numpy as np
    import faiss
    
    embeddings_path = Path(embeddings_path)
    index_path = Path(index_path)
    
    # Load embeddings
    embeddings = np.load(embeddings_path, allow_pickle=False)
    
    # Load FAISS index
    index = faiss.read_index(str(index_path))
    
    return embeddings, index


def _query_faiss_pool(
    index: Any,
    jd_embeddings,
    top_k: int = 500,
) -> set[int]:
    """
    Query FAISS index for top-K candidates per JD vector.
    
    Parameters
    ----------
    index : faiss.Index
        FAISS index (assumed flat, L2).
    jd_embeddings : np.ndarray
        Shape (n_vectors, embedding_dim).
    top_k : int
        Number of top candidates to retrieve per vector.
    
    Returns
    -------
    set
        Deduplicated candidate indices to score.
    """
    import numpy as np
    
    # Flatten to single query if needed
    if len(jd_embeddings.shape) == 1:
        jd_embeddings = jd_embeddings.reshape(1, -1)
    
    # Query FAISS for each JD vector
    distances, indices = index.search(jd_embeddings, top_k)
    
    # Deduplicate and flatten
    candidate_pool = set()
    for idx_array in indices:
        for idx in idx_array:
            if idx >= 0:  # -1 indicates not found
                candidate_pool.add(int(idx))
    
    return candidate_pool


def run_ranking_pipeline(
    candidates_path: str | Path,
    jd_path: str | Path,
    output_path: str | Path,
) -> None:
    """
    Execute the full ranking pipeline and write the submission CSV.

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
    RuntimeError
        If wall-clock time exceeds config.MAX_RANKING_WALL_CLOCK_SECONDS.
    """
    import numpy as np
    
    start_time = time.monotonic()
    print("Starting ranking pipeline...")
    
    # ===== Stage 1: Load artifacts =====
    print("Loading candidates and FAISS artifacts...")
    candidates = _load_candidates_dict(candidates_path)
    print(f"  Loaded {len(candidates)} candidates")
    
    try:
        embeddings, faiss_index = _load_faiss_artifacts(
            config.CANDIDATE_EMBEDDINGS_PATH,
            config.FAISS_INDEX_PATH,
        )
        print(f"  Loaded FAISS index with {len(embeddings)} embeddings")
    except FileNotFoundError as e:
        print(f"Error: FAISS artifacts not found: {e}")
        print("  Please run Stage 3 & 4 first to generate embeddings and index.")
        raise
    
    # Load JD embeddings
    try:
        jd_embeddings = np.load(config.JD_EMBEDDINGS_PATH, allow_pickle=False)
        print(f"  Loaded JD embeddings shape {jd_embeddings.shape}")
    except FileNotFoundError:
        print("Warning: JD embeddings not found, using neutral retrieval")
        jd_embeddings = np.zeros((1, embeddings.shape[1]), dtype=np.float32)
    
    # ===== Stage 2: FAISS retrieval =====
    print("Querying FAISS for candidate pool...")
    candidate_indices = _query_faiss_pool(faiss_index, jd_embeddings, top_k=500)
    print(f"  Retrieved {len(candidate_indices)} unique candidates")
    
    # ===== Stage 3: Score each candidate =====
    print("Scoring candidates...")
    scored_candidates: List[ScoreBreakdown] = []
    honeypot_count = 0
    error_count = 0
    
    for idx, candidate_idx in enumerate(candidate_indices):
        if idx % 100 == 0:
            print(f"  Processed {idx}/{len(candidate_indices)}")
        
        # Find candidate by index
        candidate_id = None
        for cid, candidate in candidates.items():
            # Map index to candidate (assuming order)
            # In real implementation, store index → id mapping
            if hash(cid) % len(candidates) == candidate_idx % len(candidates):
                candidate_id = cid
                break
        
        # Fallback: try to get candidate directly by ID pattern
        if candidate_id is None:
            # Create synthetic ID from index
            candidate_id = f"CAND_{candidate_idx:07d}"
        
        if candidate_id not in candidates:
            continue
        
        candidate = candidates[candidate_id]
        
        try:
            # Extract career signals
            career_signals = extract_career_signals(candidate)
            
            # Check honeypot (hard gate)
            honeypot_score, is_honeypot = compute_honeypot_score(candidate)
            if is_honeypot:
                honeypot_count += 1
                continue
            
            # Compute behavioral scores
            availability_multiplier = compute_availability_multiplier(candidate)
            behavioral_score = compute_behavioral_score(candidate)
            
            # Compute composite score
            # Use similarity scores (simplified for now)
            skill_match = 0.7  # placeholder
            career_match = 0.6  # placeholder
            disqualifier = 0.1  # placeholder
            
            breakdown = compute_composite_score(
                candidate_id=candidate_id,
                skill_match_sim=skill_match,
                career_match_sim=career_match,
                disqualifier_sim=disqualifier,
                career_signals=career_signals,
                availability_multiplier=availability_multiplier,
                behavioral_score=behavioral_score,
            )
            
            scored_candidates.append(breakdown)
        
        except Exception as e:
            error_count += 1
            if error_count <= 5:  # Log first 5 errors
                print(f"    Warning: Error scoring {candidate_id}: {e}")
            continue
    
    print(f"  Scored {len(scored_candidates)} candidates")
    print(f"  Excluded {honeypot_count} honeypots, {error_count} errors")
    
    # ===== Stage 4: Sort and select top-100 =====
    print("Selecting top 100...")
    scored_candidates.sort(key=lambda s: (-s.final_score, s.candidate_id))
    top_100 = scored_candidates[:config.TOP_K]
    print(f"  Selected {len(top_100)} for submission")
    
    # ===== Stage 5: Generate reasoning (optional) =====
    print("Generating reasoning strings...")
    rows = []
    for rank, breakdown in enumerate(top_100, start=1):
        # Simple reasoning generation
        reasoning = (
            f"Skill: {breakdown.skill_match:.2f}, "
            f"Career: {breakdown.career_match:.2f}, "
            f"Behavioral: {breakdown.behavioral_score:.2f}, "
            f"Multiplier: {breakdown.availability_multiplier:.2f}"
        )
        rows.append((breakdown.candidate_id, rank, breakdown.final_score, reasoning))
    
    # ===== Stage 6: Write CSV =====
    print(f"Writing submission to {output_path}...")
    write_submission_csv(output_path, rows)
    
    # ===== Timing validation =====
    elapsed = time.monotonic() - start_time
    print(f"Pipeline completed in {elapsed:.2f}s")
    
    if elapsed > config.MAX_RANKING_WALL_CLOCK_SECONDS:
        raise RuntimeError(
            f"Ranking exceeded time limit: {elapsed:.1f}s > "
            f"{config.MAX_RANKING_WALL_CLOCK_SECONDS}s"
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
