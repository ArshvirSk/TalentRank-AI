"""
Stage 2 — Honeypot / Fraud detection.

Owner: Member A

Checks for resume contradictions, impossible timelines, and platform
signal mismatches. Outputs a honeypot_score [0, 1] and a boolean flag
if the score crosses the configured threshold.
"""

import config
from src.features.schema import CandidateRecord


def compute_honeypot_score(candidate: CandidateRecord) -> tuple[float, bool]:
    """
    Returns (score, flag) for a candidate.
    Score is [0, 1]. Flag is True if score > HONEYPOT_FLAG_THRESHOLD.
    """
    score = 0.0

    # 1. Timeline contradiction: sum of durations > years_of_experience
    total_months = sum(e.duration_months for e in candidate.career_history)
    reported_months = candidate.years_of_experience * 12
    if total_months - reported_months > config.HONEYPOT_EXPERIENCE_TOLERANCE_MONTHS:
        score += 0.4  # strong signal of inflated resume

    # 2. Impossible expert: Claims 'expert' but duration is tiny
    for skill in candidate.skills:
        if (
            skill.proficiency.lower() == "expert"
            and skill.duration_months < config.HONEYPOT_EXPERT_MIN_DURATION_MONTHS
        ):
            score += 0.2
            break  # count once

    # 3. Platform contradiction: 100% complete profile but zero verifications
    signals = candidate.redrob_signals
    if (
        signals.profile_completeness_score >= config.HONEYPOT_COMPLETENESS_THRESHOLD
        and not signals.verified_email
        and not signals.verified_phone
        and not signals.linkedin_connected
    ):
        score += 0.3

    # Cap score at 1.0
    final_score = min(1.0, score)
    flag = final_score > config.HONEYPOT_FLAG_THRESHOLD

    return final_score, flag
