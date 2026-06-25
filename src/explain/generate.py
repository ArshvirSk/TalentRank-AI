"""
Stage 8 — Explanation Generator.

Owner: Member 4

Generates template-based reasoning strings for the submission CSV.
Slot-fills templates from extracted facts only — zero hallucination risk.

Key constraints:
  - Reasoning must reference only facts present in the candidate record.
  - Tone should match the rank (enthusiastic for top-10, matter-of-fact
    for rank 90+).
  - No two consecutive candidates should have identical reasoning text
    (use phrasing variant rotation).
"""

from __future__ import annotations

from src.ranking.fusion import ScoreBreakdown
from src.retrieval.jd_parser import JDProfile


def generate_reasoning(
    breakdown: ScoreBreakdown,
    jd_profile: JDProfile,
    rank: int,
    candidate_facts: dict | None = None,
) -> str:
    """
    Generate a human-readable reasoning string for one ranked candidate.

    Parameters
    ----------
    breakdown : ScoreBreakdown
        Full score breakdown from Stage 5.
    jd_profile : JDProfile
        Parsed JD profile (for referencing requirement categories).
    rank : int
        Candidate's rank (1-indexed).  Affects tone selection.
    candidate_facts : dict, optional
        Additional extracted facts (skill names, company names, years, etc.)
        used to fill template slots.  If None, only score-based reasoning
        is generated.

    Returns
    -------
    str
        A reasoning string suitable for the ``reasoning`` column of the
        submission CSV.  Should be a single paragraph, no newlines.

    Raises
    ------
    NotImplementedError
        Scaffold — not yet implemented.

    Notes
    -----
    Implementation checklist (Member 4):
      1. Determine tone bucket from rank:
         - ranks 1–10:  enthusiastic ("Excellent", "Outstanding")
         - ranks 11–50: positive ("Strong", "Solid")
         - ranks 51–80: neutral ("Adequate", "Meets requirements")
         - ranks 81–100: matter-of-fact ("Partially matches", "Limited")
      2. For each scoring component (skill, career, behavioral, shipping):
         a. Pick a template variant via templates.get_template_variant().
         b. Slot-fill with candidate_facts.
         c. Skip components where candidate_facts has no data.
      3. Concatenate filled segments into a single-paragraph string.
      4. Ensure the result is non-empty (fallback to neutral filler).
    """
    # TODO(Member 4): implement slot-filling explanation generation
    raise NotImplementedError(
        "generate_reasoning: explanation generation not yet implemented"
    )
