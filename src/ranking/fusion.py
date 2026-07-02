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
    
    All components in [0, 1] unless otherwise noted.
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


def _extract_availability_stability(career_signals: CareerSignals) -> float:
    """
    Compute availability_stability from career signals.
    
    This reflects job tenure stability and likelihood of staying.
    
    Parameters
    ----------
    career_signals : CareerSignals
        Career signals from Stage 2.
    
    Returns
    -------
    float
        Availability stability score in [0, 1].
        - 1.0 = very stable (long tenure, consistent history)
        - 0.5 = moderate stability
        - 0.0 = unstable (frequent job changes)
    """
    # Check if consulting_only (unstable indicator)
    if career_signals.consulting_only_flag:
        return 0.5
    
    # Use shipping_score as stability proxy (production work is more stable)
    if hasattr(career_signals, 'shipping_score'):
        shipping = float(career_signals.shipping_score)
        # Combine shipping score with flag inversions
        stability = 0.6 * shipping + 0.2 + (0.2 if not career_signals.stale_coder_flag else 0.0)
        return max(0.0, min(1.0, stability))
    
    # Default: moderate
    return 0.6


def _extract_seniority_and_shipping(career_signals: CareerSignals) -> float:
    """
    Compute seniority and shipping capability from career signals.
    
    Parameters
    ----------
    career_signals : CareerSignals
        Career signals from Stage 2.
    
    Returns
    -------
    float
        Seniority score in [0, 1].
        - 1.0 = senior (10+ years, leadership roles)
        - 0.5 = mid-level (5-10 years)
        - 0.0 = junior (0-2 years)
    """
    # Use seniority_score directly if available
    if hasattr(career_signals, 'seniority_score'):
        seniority = float(career_signals.seniority_score)
        # Also factor in shipping/production evidence
        if hasattr(career_signals, 'shipping_score'):
            shipping = float(career_signals.shipping_score)
            # Combine: 70% seniority + 30% shipping
            combined = 0.7 * seniority + 0.3 * shipping
            return max(0.0, min(1.0, combined))
        return max(0.0, min(1.0, seniority))
    
    # Fallback: use shipping_score alone
    if hasattr(career_signals, 'shipping_score'):
        return max(0.0, min(1.0, float(career_signals.shipping_score)))
    
    # Default neutral
    return 0.5


def compute_composite_score(
    candidate_id: str,
    skill_match_sim: float,
    career_match_sim: float,
    disqualifier_sim: float,
    career_signals: CareerSignals,
    availability_multiplier: float,
    behavioral_score: float = 0.5,
) -> ScoreBreakdown:
    """
    Compute the final composite score for one candidate.

    Parameters
    ----------
    candidate_id : str
        Unique candidate identifier.
    skill_match_sim : float
        Cosine similarity between candidate embedding and JD must-have vector.
        Range: [0, 1].
    career_match_sim : float
        Cosine similarity between candidate embedding and JD nice-to-have vector.
        Range: [0, 1].
    disqualifier_sim : float
        Cosine similarity between candidate embedding and JD disqualifier vector.
        Higher = worse. Range: [0, 1].
    career_signals : CareerSignals
        Extracted career signals (from Stage 2).
    availability_multiplier : float
        Multiplier derived from redrob_signals. Range: [0.6, 1.15].
    behavioral_score : float
        Behavioral score from Stage 5. Range: [0, 1]. Default: 0.5.

    Returns
    -------
    ScoreBreakdown
        Full score breakdown with all components.

    Notes
    -----
    Composite formula:
        base = 0.35*skill + 0.25*career + 0.20*behavioral + 
               0.10*stability + 0.10*seniority
        
        if disqualifier_sim > 0.6:
            base -= 0.85 * disqualifier_sim  (near-zeroing penalty)
        
        final = base * availability_multiplier
        
    All components are clamped to valid ranges for auditability.
    """
    # ===== Validate and clamp input ranges =====
    skill_match = max(0.0, min(1.0, float(skill_match_sim)))
    career_match = max(0.0, min(1.0, float(career_match_sim)))
    disqualifier = max(0.0, min(1.0, float(disqualifier_sim)))
    behavioral = max(0.0, min(1.0, float(behavioral_score)))
    multiplier = max(0.6, min(1.15, float(availability_multiplier)))
    
    # ===== Extract derived components =====
    availability_stability = _extract_availability_stability(career_signals)
    seniority_and_shipping = _extract_seniority_and_shipping(career_signals)
    
    # ===== Compute base score =====
    # Weights must sum to 1.0
    weights = {
        'skill_match': 0.45,
        'career_match': 0.20,
        'behavioral_score': 0.15,
        'availability_stability': 0.10,
        'seniority_and_shipping': 0.10,
    }
    
    total_weight = sum(weights.values())
    assert abs(total_weight - 1.0) < 0.01, f"Weights must sum to 1.0, got {total_weight}"
    
    base_score = (
        weights['skill_match'] * skill_match +
        weights['career_match'] * career_match +
        weights['behavioral_score'] * behavioral +
        weights['availability_stability'] * availability_stability +
        weights['seniority_and_shipping'] * seniority_and_shipping
    )
    
    base_score = max(0.0, min(1.0, base_score))
    
    # ===== Apply disqualifier penalty =====
    # If disqualifier_sim > 0.75, apply heavy penalty to near-zero
    DISQUALIFIER_THRESHOLD = 0.75
    DISQUALIFIER_PENALTY_SCALE = 0.85
    
    disqualifier_penalty = 0.0
    if disqualifier > DISQUALIFIER_THRESHOLD:
        disqualifier_penalty = DISQUALIFIER_PENALTY_SCALE * disqualifier
        base_score = max(0.0, base_score - disqualifier_penalty)
    
    # ===== Apply availability multiplier =====
    final_score = base_score * multiplier
    
    # ===== Clamp final score to [0, 1] =====
    final_score = max(0.0, min(1.0, final_score))
    
    # ===== Return full breakdown =====
    return ScoreBreakdown(
        candidate_id=candidate_id,
        skill_match=skill_match,
        career_match=career_match,
        behavioral_score=behavioral,
        availability_stability=availability_stability,
        seniority_and_shipping=seniority_and_shipping,
        disqualifier_penalty=disqualifier_penalty,
        availability_multiplier=multiplier,
        base_score=base_score,
        final_score=final_score,
    )
