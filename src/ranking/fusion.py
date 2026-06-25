"""
Stage 5 — Hybrid Candidate Scoring (fusion).

Owner: Member 3

Computes the composite score for each candidate:

    base = 0.35 * skill_match
         + 0.25 * career_match
         + 0.20 * behavioral_score
         + 0.10 * availability_stability
         + 0.10 * seniority_and_shipping_score

    base -= disqualifier_penalty   # heavy, near-zeroing

    final = base * availability_multiplier   # ~0.6–1.15 from redrob_signals

skill_match and career_match are derived from Stage 4 (FAISS similarity).
behavioral_score and availability_multiplier are derived from Stage 2
(career signals and redrob_signals).
"""

from __future__ import annotations

from dataclasses import dataclass

import config
from src.features.career_signals import CareerSignals


@dataclass
class ScoreBreakdown:
    """
    Full breakdown of a candidate's composite score for auditability.
    """
    candidate_id: str = ""
    skill_match: float = 0.0
    career_match: float = 0.0
    behavioral_score: float = 0.0
    availability_stability: float = 0.0
    seniority_and_shipping: float = 0.0
    disqualifier_penalty: float = 0.0
    availability_multiplier: float = 1.0
    base_score: float = 0.0
    final_score: float = 0.0

    # TODO(Member 3): add any additional diagnostic fields


def compute_composite_score(
    candidate_id: str,
    skill_match_sim: float,
    career_match_sim: float,
    disqualifier_sim: float,
    career_signals: CareerSignals,
    availability_multiplier: float,
) -> ScoreBreakdown:
    """
    Compute the final composite score for one candidate.

    Parameters
    ----------
    candidate_id : str
        Unique candidate identifier.
    skill_match_sim : float
        Cosine similarity between candidate embedding and JD must-have vector
        (from Stage 4).
    career_match_sim : float
        Cosine similarity between candidate embedding and JD nice-to-have
        vector (from Stage 4).
    disqualifier_sim : float
        Cosine similarity between candidate embedding and JD disqualifier
        vector (from Stage 4).  Higher = worse.
    career_signals : CareerSignals
        Extracted career signals (from Stage 2).
    availability_multiplier : float
        Multiplier derived from redrob_signals (from behavioral.py).

    Returns
    -------
    ScoreBreakdown
        Full score breakdown.

    Raises
    ------
    NotImplementedError
        Scaffold — not yet implemented.

    Notes
    -----
    Implementation checklist (Member 3):
      1. Normalize similarity scores to [0, 1] range if needed.
      2. Compute each weighted component.
      3. Apply disqualifier_penalty: if disqualifier_sim > threshold,
         subtract a heavy penalty (config.DISQUALIFIER_PENALTY_SCALE).
      4. Multiply by availability_multiplier.
      5. Clamp final_score to [0, 1].
      6. Return full ScoreBreakdown for auditability.
    """
    # TODO(Member 3): implement composite scoring formula
    raise NotImplementedError(
        "compute_composite_score: scoring formula not yet implemented"
    )
