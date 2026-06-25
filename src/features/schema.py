"""
Stage 2 (partial) — Typed schema for candidate_schema.json fields.

Owner: Member 1

Parses raw JSONL candidate records into typed Python dataclasses so that
downstream stages can rely on field presence and types without defensive
checks everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Typed sub-records
# ---------------------------------------------------------------------------

@dataclass
class SkillEntry:
    """A single skill from the candidate's skills list."""
    name: str = ""
    proficiency: str = ""           # e.g. "beginner", "intermediate", "expert"
    endorsements: int = 0
    duration_months: int = 0

    # TODO(Member 1): add any additional fields present in candidate_schema.json


@dataclass
class CareerEntry:
    """A single position from the candidate's career_history."""
    title: str = ""
    company: str = ""
    start_date: str = ""            # ISO date string
    end_date: str = ""              # ISO date string or "present"
    duration_months: int = 0
    description: str = ""

    # TODO(Member 1): add sector/industry classification if available


@dataclass
class EducationEntry:
    """A single education record."""
    degree: str = ""
    institution: str = ""
    field_of_study: str = ""
    start_year: int = 0
    end_year: int = 0

    # TODO(Member 1): add graduation flag, GPA if present


@dataclass
class RedrobSignal:
    """A single entry from the redrob_signals array."""
    signal_name: str = ""
    value: Any = None

    # TODO(Member 1): enumerate known signal names as constants


@dataclass
class CandidateRecord:
    """
    Fully typed representation of one candidate from candidates.jsonl.

    This is the canonical in-memory representation consumed by all pipeline
    stages.  Raw JSON dicts should be converted to this type via
    ``parse_candidate()`` as early as possible.
    """
    candidate_id: str = ""
    years_of_experience: float = 0.0
    skills: list[SkillEntry] = field(default_factory=list)
    career_history: list[CareerEntry] = field(default_factory=list)
    education: list[EducationEntry] = field(default_factory=list)
    redrob_signals: list[RedrobSignal] = field(default_factory=list)
    location: str = ""
    summary: str = ""

    # TODO(Member 1): map remaining top-level fields from the schema


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_candidate(raw: dict[str, Any]) -> CandidateRecord:
    """
    Convert a raw JSON dict (one line from candidates.jsonl) into a
    ``CandidateRecord``.

    Parameters
    ----------
    raw : dict
        Deserialized JSON object for a single candidate.

    Returns
    -------
    CandidateRecord
        Typed, validated candidate record.

    Raises
    ------
    NotImplementedError
        Scaffold — not yet implemented.
    """
    # TODO(Member 1): implement field mapping with sensible defaults for
    # missing/null fields.  Validate candidate_id is non-empty.
    raise NotImplementedError("parse_candidate: field mapping not yet implemented")
