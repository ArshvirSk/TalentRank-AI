"""
Stage 8 (partial) — Phrasing variant bank for explanation generation.

Owner: Member 4

Contains multiple phrasing templates per JD-aspect bucket so that
100 reasoning strings are not templated-identical.  Templates are
slot-filled only with facts present in the candidate record
(no hallucination risk).

Format: compact, semicolon-separated clauses — not full prose sentences.
Each template produces one clause only.  The generator assembles clauses
into a single semicolon-joined string.

Template categories:
  - opening       (rank + score + skill clause)
  - skill_clause  (with data, or scores-only fallback)
  - career        (with data, or scores-only fallback)
  - behavioral    (with data, or scores-only fallback)
  - disqualifier / honeypot / no-penalty
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Opening — rank + score + pre-built skill_clause slot
# ---------------------------------------------------------------------------

OPENING_TEMPLATES: list[str] = [
    "Rank {rank} ({score:.3f}): {skill_clause}",
    "#{rank} of 100 | score {score:.3f} -- {skill_clause}",
    "Composite {score:.3f} (rank {rank}): {skill_clause}",
    "Score {score:.3f}, rank {rank} -- {skill_clause}",
    "Final {score:.3f} | rank {rank} | {skill_clause}",
]

# ---------------------------------------------------------------------------
# Skill clause — when candidate_facts has top_skills
# ---------------------------------------------------------------------------

SKILL_CLAUSE_WITH_DATA: list[str] = [
    "{match_count}/{total_count} JD skills matched; top: {skills_with_months}",
    "skill match {skill_match:.2f}; {match_count}/{total_count} must-haves; {skills_with_months}",
    "{match_count} of {total_count} core skills present ({skills_with_months})",
    "JD skill coverage {match_count}/{total_count}; leading: {skills_with_months}",
    "skill sim {skill_match:.2f}; matched {match_count}/{total_count} -- {skills_with_months}",
]

# ---------------------------------------------------------------------------
# Skill clause — fallback when no top_skills in facts
# ---------------------------------------------------------------------------

SKILL_CLAUSE_SCORES_ONLY: list[str] = [
    "skill sim {skill_match:.2f}",
    "JD skill similarity {skill_match:.2f}",
    "must-have skill score {skill_match:.2f}",
    "skill coverage score {skill_match:.2f}",
    "skill match {skill_match:.2f}",
]

# ---------------------------------------------------------------------------
# Career — when years / roles / current_role_months available
# ---------------------------------------------------------------------------

CAREER_TEMPLATES: list[str] = [
    "{years}y across {roles} roles; current {current_role_months}mo",
    "{years}y experience; {roles} employers; {current_role_months}mo in current role",
    "career: {years}y / {roles} roles / {current_role_months}mo tenure",
    "{years}y background; {roles} positions; {current_role_months}mo at current firm",
    "{years}y exp ({roles} roles); {current_role_months}mo current tenure",
]

# ---------------------------------------------------------------------------
# Career — fallback scores-only
# ---------------------------------------------------------------------------

CAREER_SCORES_ONLY_TEMPLATES: list[str] = [
    "career sim {career_match:.2f}; seniority+shipping {seniority:.2f}",
    "experience match {career_match:.2f}; shipping score {seniority:.2f}",
    "career score {career_match:.2f} | seniority {seniority:.2f}",
    "career alignment {career_match:.2f}; delivery signal {seniority:.2f}",
    "career {career_match:.2f} / seniority {seniority:.2f}",
]

# ---------------------------------------------------------------------------
# Behavioral — when response_rate or notice_period_days available
# ---------------------------------------------------------------------------

BEHAVIORAL_TEMPLATES: list[str] = [
    "response rate {response_rate}; notice {notice_period}d; {work_status}",
    "recruiter response {response_rate}; {notice_period}d notice; {work_status}",
    "availability mult {avail_mult:.2f}; response {response_rate}; {notice_period}d notice",
    "behavioral {behavioral:.2f}; response {response_rate}; notice {notice_period}d",
    "avail score {avail_stability:.2f}; response {response_rate}; {work_status}",
]

# ---------------------------------------------------------------------------
# Behavioral — fallback scores-only
# ---------------------------------------------------------------------------

BEHAVIORAL_SCORES_ONLY_TEMPLATES: list[str] = [
    "behavioral {behavioral:.2f}; avail mult {avail_mult:.2f}",
    "availability score {avail_stability:.2f}; behavioral {behavioral:.2f}",
    "avail mult {avail_mult:.2f}; behavioral score {behavioral:.2f}",
    "behavioral signal {behavioral:.2f}; availability {avail_stability:.2f}",
    "avail {avail_stability:.2f} / behavioral {behavioral:.2f}",
]

# ---------------------------------------------------------------------------
# Disqualifier — when disqualifier_flag + areas present, or penalty > 0
# ---------------------------------------------------------------------------

DISQUALIFIER_TEMPLATES: list[str] = [
    "DISQUALIFIER HIT: {areas}; penalty {penalty:.2f} applied",
    "disqualifier penalty {penalty:.2f} ({areas})",
    "flagged: {areas}; score reduced by {penalty:.2f}",
    "disqualifier: {areas} -- penalty {penalty:.2f}",
    "penalty {penalty:.2f} for: {areas}",
]

# ---------------------------------------------------------------------------
# Honeypot — plain strings, no format slots
# ---------------------------------------------------------------------------

HONEYPOT_TEMPLATES: list[str] = [
    "internal consistency flag raised",
    "profile integrity check failed",
    "automated integrity check flagged this profile",
    "signal inconsistency detected in profile",
    "honeypot signals present",
]

# ---------------------------------------------------------------------------
# No penalty — plain strings, no format slots
# ---------------------------------------------------------------------------

NO_PENALTY_TEMPLATES: list[str] = [
    "no disqualifiers",
    "0 disqualifier hits",
    "clean disqualifier check",
    "disqualifier penalty: none",
    "no flags raised",
]

# ---------------------------------------------------------------------------
# Gap acknowledgment — fires for rank >= BORDERLINE_RANK_THRESHOLD or
# final_score < BORDERLINE_SCORE_THRESHOLD (see config.py).
# Slots available: {rank}, {skill_match:.2f}, {top_gap}
# All variants must be safe to .format(rank=..., skill_match=..., top_gap=...)
# ---------------------------------------------------------------------------

GAP_ACKNOWLEDGMENT_TEMPLATES: list[str] = [
    "partial match only -- {top_gap}",
    "limited alignment with JD core requirements; {top_gap}",
    "borderline inclusion (rank {rank} of 100); {top_gap}",
    "weak JD fit; {top_gap}",
    "gap acknowledged: skill sim {skill_match:.2f}; {top_gap}",
]


# ---------------------------------------------------------------------------
# Variant selector — deterministic, no random
# ---------------------------------------------------------------------------

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
        Selected template string (not yet slot-filled).
    """
    index = hash(candidate_id + str(rank)) % len(templates)
    return templates[index]
