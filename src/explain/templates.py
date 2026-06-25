"""
Stage 8 (partial) — Phrasing variant bank for explanation generation.

Owner: Member 4

Contains multiple phrasing templates per JD-aspect bucket so that
100 reasoning strings are not templated-identical.  Templates are
slot-filled only with facts present in the candidate record
(no hallucination risk).

Template categories mirror the scoring components:
  - skill_match
  - career_match
  - behavioral / availability
  - seniority / shipping
  - disqualifier (negative framing for lower ranks)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Template bank — each key maps to a list of phrasing variants.
# The generator picks a variant based on (rank % len(variants)) or a hash
# of candidate_id to ensure variety without randomness.
# ---------------------------------------------------------------------------

# TODO(Member 4): expand each list to 4–6 variants with natural-sounding
# phrasing.  Tone should degrade gracefully: top-10 candidates get
# enthusiastic language, ranks 50+ get neutral, ranks 90+ are matter-of-fact.

SKILL_MATCH_TEMPLATES: list[str] = [
    "Demonstrates strong alignment with required skills including {skills}.",
    "Skill profile covers {match_count} of {total_count} must-have competencies ({skills}).",
    "Possesses key technical skills: {skills}.",
]

CAREER_MATCH_TEMPLATES: list[str] = [
    "Career trajectory shows {years} years of relevant experience in {domains}.",
    "Professional background in {domains} aligns with the role's requirements.",
    "Has built {years} years of experience across {domains}.",
]

SHIPPING_TEMPLATES: list[str] = [
    "Has evidence of shipping production systems at {companies}.",
    "Track record includes deploying production-grade solutions at {companies}.",
    "Demonstrated ability to deliver production systems ({companies}).",
]

BEHAVIORAL_TEMPLATES: list[str] = [
    "Availability signals indicate {availability_status}.",
    "Currently {availability_status} with {notice_period} notice period.",
]

DISQUALIFIER_TEMPLATES: list[str] = [
    "Profile shows partial overlap with disqualified areas ({areas}), reducing overall fit.",
    "Some experience in {areas} may not directly transfer to this role.",
]

NEUTRAL_FILLER_TEMPLATES: list[str] = [
    "Ranked based on composite evaluation of skills, experience, and availability.",
    "Overall profile assessed across technical skills, career alignment, and behavioural signals.",
]


def get_template_variant(
    templates: list[str],
    candidate_id: str,
    rank: int,
) -> str:
    """
    Select a phrasing variant deterministically based on candidate_id and rank.

    Parameters
    ----------
    templates : list[str]
        List of template strings with {slot} placeholders.
    candidate_id : str
        Used for deterministic variant selection.
    rank : int
        Current rank (1-indexed).

    Returns
    -------
    str
        Selected template string (not yet filled).
    """
    # TODO(Member 4): implement deterministic variant selection
    # Use hash(candidate_id + str(rank)) % len(templates) for variety
    raise NotImplementedError(
        "get_template_variant: variant selection not yet implemented"
    )
