"""
Stage 2 (partial) — Trust-weighted skill scoring.

Owner: Member 1

Computes a trust-adjusted skill relevance score per candidate:

    trust_score = Σ (endorsements × duration_months × proficiency_weight)
                  for skills that pass cross-validation against career_history

Skills that are *listed* but never mentioned in any career_history description
are discounted (the candidate may have added them as keyword padding).
"""

from __future__ import annotations

from src.features.schema import CandidateRecord


# Proficiency level → numeric weight mapping
PROFICIENCY_WEIGHTS: dict[str, float] = {
    "beginner": 0.2,
    "intermediate": 0.5,
    "advanced": 0.8,
    "expert": 1.0,
}


def compute_skill_trust_score(
    candidate: CandidateRecord,
    must_have_skills: list[str] | None = None,
    nice_to_have_skills: list[str] | None = None,
) -> float:
    """
    Compute a trust-weighted skill relevance score for a single candidate.

    The formula per skill is:
        endorsements × duration_months × proficiency_weight

    Skills are cross-validated against career_history text — if a listed skill
    never appears in any job description, its contribution is discounted by a
    configurable factor.

    Parameters
    ----------
    candidate : CandidateRecord
        Typed candidate record.
    must_have_skills : list[str], optional
        Skills from the JD's must-have block.  If provided, only matching
        skills contribute to the score.
    nice_to_have_skills : list[str], optional
        Skills from the JD's nice-to-have block.  Matching skills contribute
        at a reduced weight.

    Returns
    -------
    float
        Aggregated trust-weighted skill score (non-negative).

    Raises
    ------
    NotImplementedError
        Scaffold — not yet implemented.

    Notes
    -----
    Implementation checklist (Member 1):
      1. For each skill in candidate.skills:
         a. Look up proficiency_weight from PROFICIENCY_WEIGHTS.
         b. Check if skill.name appears in any career_history[].description
            (case-insensitive, consider stemming/aliases).
         c. If cross-validated: full weight.
            If not cross-validated: apply a discount factor (e.g. 0.3).
         d. Accumulate: endorsements * duration_months * proficiency_weight
                         * cross_validation_factor
      2. Optionally weight must-have matches higher than nice-to-have.
      3. Normalize or return raw sum (TBD with Member 3).
    """
    # TODO(Member 1): implement trust-weighted skill scoring
    raise NotImplementedError(
        "compute_skill_trust_score: scoring logic not yet implemented"
    )
