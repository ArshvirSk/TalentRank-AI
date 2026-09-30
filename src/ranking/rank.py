"""
Stage 7 — Final Ranking Engine (CLI entrypoint).

Owner: Member 3 (integrated by team lead)

Loads precomputed artifacts (embeddings, candidate index, JD query vectors,
feature parquet), computes real cosine similarities, applies Stage 5
composite scoring + Stage 6 honeypot exclusion, and writes the top-100 CSV.

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
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

import config
from src.retrieval import jd_parser as jd_parser_module
from src.features.schema import CandidateRecord, parse_candidate
from src.features.career_signals import extract_career_signals
from src.features.honeypot import compute_honeypot_score
from src.ranking.behavioral import compute_availability_multiplier, compute_behavioral_score
from src.ranking.fusion import compute_composite_score, ScoreBreakdown
from src.retrieval.jd_parser import JDProfile, load_default_jd_profile
from src.retrieval import faiss_index as retrieval

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_candidates_parsed(candidates_path: str | Path) -> Dict[str, CandidateRecord]:
    """Load candidates.jsonl into a dict keyed by candidate_id (parsed CandidateRecords)."""
    candidates = {}
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                record = parse_candidate(raw)
                candidates[record.candidate_id] = record
            except Exception:
                continue
    return candidates


def _load_precomputed_artifacts() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load Member B's precomputed artifacts:
      - candidate_embeddings.npy  (100000, 384) float32, L2-normalized
      - jd_query_vectors.npy      (3, 384) float32 — [must, nice, disq]
      - candidate_index.parquet   [candidate_id, idx]

    Returns (embeddings, jd_vectors, index_df)
    """
    embeddings = np.load(config.CANDIDATE_EMBEDDINGS_PATH, allow_pickle=False)
    jd_vectors = np.load(config.JD_EMBEDDINGS_PATH, allow_pickle=False)
    index_df = pd.read_parquet(config.ARTIFACTS_DIR / "candidate_index.parquet")
    return embeddings, jd_vectors, index_df


def _load_feature_parquet() -> pd.DataFrame:
    """Load Member A's precomputed feature parquet."""
    return pd.read_parquet(config.CANDIDATE_FEATURES_PATH)


# ---------------------------------------------------------------------------
# Similarity computation
# ---------------------------------------------------------------------------

def _compute_all_similarities(
    embeddings: np.ndarray,
    jd_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute cosine similarities between all candidate embeddings and the
    3 JD query vectors (must-have, nice-to-have, disqualifier).

    Backend selected by config.RETRIEVAL_BACKEND:
      - "numpy": single vectorized matmul over all N rows — exact and, at
        100k x 384 (~150 MB), faster than index build+query. Default.
      - "faiss": exact IndexFlatIP via src/retrieval/faiss_index.py —
        identical results; the right choice at 10M+ rows or when the
        matrix is memory-mapped. Requires faiss-cpu.

    Returns (sim_must, sim_nice, sim_disq) each of shape (N,).
    """
    if config.RETRIEVAL_BACKEND == "faiss":
        from src.retrieval.faiss_index import build_faiss_index, query_faiss_index

        # FAISS handles the retrieval funnel (must-have top-k shortlist).
        # nice/disqualifier similarities are then computed EXACTLY for the
        # shortlisted rows only — fetching them from a top-k query would
        # silently return 0.0 for candidates outside a query's top-k and
        # lose disqualifier penalties. 500 rows x 384 dims is microseconds.
        index_path = build_faiss_index(embeddings=embeddings)
        k_must = min(config.FAISS_TOP_K_RETRIEVAL, embeddings.shape[0])
        d_must, i_must = query_faiss_index(
            jd_vectors[0:1], top_k=k_must, index_path=index_path
        )

        n = embeddings.shape[0]
        sim_must = np.zeros(n, dtype=np.float64)
        sim_nice = np.zeros(n, dtype=np.float64)
        sim_disq = np.zeros(n, dtype=np.float64)

        shortlist = i_must[0]
        sim_must[shortlist] = d_must[0]
        short_rows = embeddings[shortlist]
        sim_nice[shortlist] = short_rows @ jd_vectors[1]
        sim_disq[shortlist] = short_rows @ jd_vectors[2]
        return sim_must, sim_nice, sim_disq

    # Default: numpy matmul backend (exact)
    # jd_vectors[0] = must-have, [1] = nice-to-have, [2] = disqualifier
    sim_must = embeddings @ jd_vectors[0]   # (N,)
    sim_nice = embeddings @ jd_vectors[1]   # (N,)
    sim_disq = embeddings @ jd_vectors[2]   # (N,)

    # Clamp to [0, 1] (cosine sim of normalized vectors is in [-1, 1])
    sim_must = np.clip(sim_must, 0.0, 1.0)
    sim_nice = np.clip(sim_nice, 0.0, 1.0)
    sim_disq = np.clip(sim_disq, 0.0, 1.0)

    return sim_must, sim_nice, sim_disq


# ---------------------------------------------------------------------------
# Core ranking pipeline
# ---------------------------------------------------------------------------

def run_ranking_pipeline(
    candidates_path: str | Path,
    jd_path: str | Path,
    output_path: str | Path,
) -> None:
    """
    Execute the full ranking pipeline and write the submission CSV.

    Steps:
      1. Load precomputed artifacts (embeddings, JD vectors, feature parquet)
      2. Compute cosine similarities for all 100k candidates (vectorized)
      3. Pre-filter: take top-500 by sim_must (FAISS-equivalent shortlist)
      4. Score each shortlisted candidate using the composite formula
      5. Sort and select top-100
      6. Generate reasoning via Member D's explainer
      7. Write CSV
    """
    start_time = time.monotonic()
    log.info("Starting ranking pipeline...")

    # ===== Step 1: Load all artifacts =====
    log.info("Loading candidates...")
    candidates = _load_candidates_parsed(candidates_path)
    log.info(f"  Loaded {len(candidates)} candidates")

    log.info("Loading precomputed embeddings and JD vectors...")
    embeddings, jd_vectors, index_df = _load_precomputed_artifacts()
    log.info(f"  Embeddings: {embeddings.shape}, JD vectors: {jd_vectors.shape}")

    # Build index→candidate_id lookup
    idx_to_cid = dict(zip(index_df["idx"], index_df["candidate_id"]))

    log.info("Loading feature parquet...")
    features_df = _load_feature_parquet()
    features_lookup = features_df.set_index("candidate_id").to_dict("index")
    log.info(f"  Loaded features for {len(features_lookup)} candidates")

    # ===== Step 2: Compute similarities (vectorized — instant) =====
    log.info("Computing cosine similarities...")
    sim_must, sim_nice, sim_disq = _compute_all_similarities(embeddings, jd_vectors)
    log.info(f"  sim_must range: [{sim_must.min():.4f}, {sim_must.max():.4f}]")
    log.info(f"  sim_nice range: [{sim_nice.min():.4f}, {sim_nice.max():.4f}]")
    log.info(f"  sim_disq range: [{sim_disq.min():.4f}, {sim_disq.max():.4f}]")

    # ===== Step 3: Pre-filter top-500 by must-have similarity =====
    retrieval_k = config.FAISS_TOP_K_RETRIEVAL  # 500
    top_indices = np.argsort(sim_must)[::-1][:retrieval_k]
    log.info(f"  Shortlisted {len(top_indices)} candidates by sim_must")

    # ===== Step 4: Score each shortlisted candidate =====
    log.info("Scoring shortlisted candidates...")
    scored_candidates: List[ScoreBreakdown] = []
    honeypot_count = 0
    error_count = 0

    for rank_pos, arr_idx in enumerate(top_indices):
        arr_idx = int(arr_idx)
        candidate_id = idx_to_cid.get(arr_idx)
        if candidate_id is None or candidate_id not in candidates:
            error_count += 1
            continue

        candidate_raw = candidates[candidate_id]

        try:
            # Get precomputed feature flags from Member A's parquet
            feat = features_lookup.get(candidate_id, {})

            # Honeypot hard gate (use precomputed flag)
            if feat.get("honeypot_flag", False):
                honeypot_count += 1
                continue

            # Career signals for fusion scoring
            career_signals = extract_career_signals(candidate_raw)

            # Behavioral scores from Member C
            availability_multiplier = compute_availability_multiplier(candidate_raw)
            behavioral_score = compute_behavioral_score(candidate_raw)

            # Use REAL similarity scores from Member B's embeddings
            skill_match = float(sim_must[arr_idx])
            career_match = float(sim_nice[arr_idx])
            disqualifier = float(sim_disq[arr_idx])

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
            if error_count <= 5:
                log.warning(f"Error scoring {candidate_id}: {e}")
            continue

    log.info(f"  Scored {len(scored_candidates)} candidates")
    log.info(f"  Excluded {honeypot_count} honeypots, {error_count} errors/missing")

    # ===== Step 5: Sort and select top-100 =====
    log.info("Selecting top 100...")
    scored_candidates.sort(key=lambda s: (-s.final_score, s.candidate_id))
    top_100 = scored_candidates[: config.TOP_K]
    log.info(f"  Selected {len(top_100)} for submission")

    # ===== Step 6: Generate reasoning strings =====
    log.info("Generating reasoning strings...")

    # Real JDProfile for Member D's explainer: prefer parsing the actual JD
    # file; fall back to the canonical hand-decomposed competition JD.
    try:
        jd_profile: JDProfile = load_default_jd_profile()
        try:
            parsed = jd_parser_module.parse_job_description(jd_path)
            if parsed.must_have_skills:
                jd_profile = parsed
                log.info("  JD parsed from file: %d must-have, %d nice-to-have, %d disqualifiers",
                         len(parsed.must_have_skills), len(parsed.nice_to_have_skills), len(parsed.disqualifiers))
            else:
                log.info("  JD file parsed but empty sections; using canonical JD profile")
        except Exception as parse_error:
            log.info("  JD file parse unavailable (%s); using canonical JD profile", parse_error)
    except Exception as fallback_error:  # pragma: no cover - defensive
        jd_profile = JDProfile()
        log.warning("  JD profile unavailable (%s); explainer runs score-only", fallback_error)

    rows = []
    for rank, breakdown in enumerate(top_100, start=1):
        # Gather facts from Member A's features for the explainer
        feat = features_lookup.get(breakdown.candidate_id, {})
        candidate_facts = {
            "honeypot_flag": feat.get("honeypot_flag", False),
            "recent_llm_only_flag": feat.get("recent_llm_only_flag", False),
            "consulting_only_flag": feat.get("consulting_only_flag", False),
            "title_chaser_score": feat.get("title_chaser_score", 0.0),
            "stale_coder_flag": feat.get("stale_coder_flag", False),
            "research_only_flag": feat.get("research_only_flag", False),
            "production_evidence_score": feat.get("production_evidence_score", 0.0),
        }

        try:
            from src.explain.generate import generate_reasoning
            reasoning = generate_reasoning(
                breakdown=breakdown,
                jd_profile=jd_profile,
                rank=rank,
                candidate_facts=candidate_facts,
            )
        except Exception:
            # Fallback if explainer fails
            reasoning = (
                f"Skill: {breakdown.skill_match:.2f}; "
                f"Career: {breakdown.career_match:.2f}; "
                f"Behavioral: {breakdown.behavioral_score:.2f}; "
                f"Score: {breakdown.final_score:.4f}"
            )

        rows.append((breakdown.candidate_id, rank, round(breakdown.final_score, 6), reasoning))

    # ===== Step 7: Write output =====
    log.info(f"Writing submission to {output_path}...")
    write_submission(output_path, rows)

    # ===== Timing validation =====
    elapsed = time.monotonic() - start_time
    log.info(f"Pipeline completed in {elapsed:.2f}s")

    if elapsed > config.MAX_RANKING_WALL_CLOCK_SECONDS:
        raise RuntimeError(
            f"Ranking exceeded time limit: {elapsed:.1f}s > "
            f"{config.MAX_RANKING_WALL_CLOCK_SECONDS}s"
        )


def write_submission(
    output_path: str | Path,
    rows: list[tuple[str, int, float, str]],
) -> None:
    """Write the final submission file (CSV or XLSX)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows, columns=["candidate_id", "rank", "score", "reasoning"])
    
    if output_path.suffix.lower() == '.xlsx':
        df.to_excel(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

    log.info(f"Submission written to {output_path} ({len(rows)} rows)")


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
        "--candidates", type=str, default=str(config.CANDIDATES_PATH),
        help="Path to candidates.jsonl",
    )
    parser.add_argument(
        "--jd", type=str, default=str(config.JD_PATH),
        help="Path to job_description.docx",
    )
    parser.add_argument(
        "--out", type=str, default=str(config.SUBMISSION_PATH),
        help="Output file path (default: submission.xlsx)",
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
