"""
Stage 6 — Honeypot Detection Layer.

Owner: Member 1

Rule-based internal-consistency checks that identify fabricated or
inconsistent candidate profiles.  Honeypots are **excluded** from the
top-100 regardless of score — this is a hard gate, not a score input.

Checks:
  - skill proficiency="expert" with duration_months ≈ 0
  - years_of_experience vs Σ(career_history.duration_months) mismatch
  - education timeline impossibilities (degree before age 16, overlapping
    full-time degrees, etc.)
  - suspiciously identical career descriptions (copy-paste padding)
"""

from __future__ import annotations

from dataclasses import dataclass
from src.features.schema import CandidateRecord


@dataclass
class HoneypotResult:
    """
    Result of honeypot detection for one candidate.

    Attributes
    ----------
    is_honeypot : bool
        True if the candidate fails any consistency check.
    reasons : list[str]
        Human-readable descriptions of each triggered rule.
    """
    is_honeypot: bool = False
    reasons: list[str] | None = None

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []


def detect_honeypot(candidate: CandidateRecord) -> HoneypotResult:
    """
    Run all consistency checks against a single candidate record.

    Parameters
    ----------
    candidate : CandidateRecord
        Typed candidate record.

    Returns
    -------
    HoneypotResult
        Detection result with flag and triggered-rule explanations.

    Raises
    ------
    NotImplementedError
        Scaffold — not yet implemented.

    Notes
    -----
    Implementation checklist (Member 1):
      1. Expert-with-zero-duration: for each skill where
         proficiency == "expert" and duration_months <= 1, flag.
      2. Experience mismatch: |years_of_experience * 12 -
         sum(career_history.duration_months)| > threshold (e.g. 24 months).
      3. Education impossibilities:
         a. Degree end_year - start_year < minimum for degree type.
         b. Overlapping full-time degrees.
         c. Bachelor's completed before age ~16 (if birth year inferrable).
      4. Duplicate descriptions: if >=3 career entries share >80% text
         similarity (Jaccard on word sets).
      5. Return HoneypotResult with all triggered reasons.
    """
    # TODO(Member 1): implement honeypot consistency checks
    raise NotImplementedError(
        "detect_honeypot: consistency checks not yet implemented"
    )
