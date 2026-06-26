"""
Stage 2 — Skill trust scoring.

Owner: Member A

Computes a trust-weighted score for each skill listed by the candidate.
Filters out obvious padding and outputs a flattened table:
(candidate_id, skill_name, trust_score).
"""

import config
from src.features.schema import CandidateRecord


def compute_skill_trust(
    proficiency: str,
    endorsements: int,
    duration_months: int,
    mentioned_in_text: bool,
) -> float:
    """
    Compute a trust score in [0, 1] for a single skill.

    Base score comes from stated proficiency. It's scaled by duration
    and endorsements. If the skill isn't mentioned in the profile/career
    text, and has no endorsements/duration, it gets severely discounted.
    """
    base_weight = config.PROFICIENCY_WEIGHTS.get(proficiency.lower(), 0.1)

    # Normalize duration (0 to cap -> 0.0 to 1.0)
    dur_factor = min(1.0, duration_months / config.SKILL_DURATION_CAP_MONTHS)

    # Normalize endorsements (0 to cap -> 0.0 to 1.0)
    end_factor = min(1.0, endorsements / config.SKILL_ENDORSEMENT_CAP)

    # Combine signals: self-reported base is trusted more if backed by time/others
    # Formula: base * (1 + 0.5 * dur + 0.5 * end)  [max multiplier is 2x]
    raw_score = base_weight * (1.0 + 0.5 * dur_factor + 0.5 * end_factor)
    trust_score = min(1.0, raw_score)

    # Discount obvious padding
    if not mentioned_in_text and endorsements == 0 and duration_months < 3:
        trust_score *= config.SKILL_PADDING_DISCOUNT

    return trust_score


def get_skill_trust_table(candidate: CandidateRecord) -> list[dict]:
    """
    Process all skills for a candidate and return a list of rows
    ready for Parquet export.
    """
    if not candidate.skills:
        return []

    # Combine all text fields to check for mentions
    all_text_parts = [candidate.headline, candidate.summary]
    for entry in candidate.career_history:
        all_text_parts.append(entry.description)
        all_text_parts.append(entry.title)

    combined_text = " ".join(all_text_parts).lower()

    rows = []
    for skill in candidate.skills:
        name_lower = skill.name.lower()
        mentioned = name_lower in combined_text

        trust_score = compute_skill_trust(
            proficiency=skill.proficiency,
            endorsements=skill.endorsements,
            duration_months=skill.duration_months,
            mentioned_in_text=mentioned,
        )

        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "skill_name": skill.name,
                "trust_score": float(trust_score),
            }
        )

    return rows
