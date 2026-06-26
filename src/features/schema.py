"""
Stage 2 (foundation) — Typed schema for candidate records.

Owner: Member A

Parses raw JSONL candidate records into typed Python dataclasses matching
the candidate_schema.json specification exactly.  Provides a streaming
``load_candidates()`` iterator that yields parsed objects one at a time
(100k records — never loads the whole file into memory).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


# ---------------------------------------------------------------------------
# Sub-record dataclasses
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SkillEntry:
    """A single skill from the candidate's skills list."""
    name: str = ""
    proficiency: str = ""           # beginner | intermediate | advanced | expert
    endorsements: int = 0
    duration_months: int = 0


@dataclass(slots=True)
class CareerEntry:
    """A single position from the candidate's career_history."""
    company: str = ""
    title: str = ""
    start_date: str = ""            # ISO date string e.g. "2021-03-15"
    end_date: str = ""              # ISO date string or empty/null for current
    duration_months: int = 0
    is_current: bool = False
    industry: str = ""
    company_size: str = ""
    description: str = ""


@dataclass(slots=True)
class EducationEntry:
    """A single education record."""
    institution: str = ""
    degree: str = ""
    field_of_study: str = ""
    start_year: int = 0
    end_year: int = 0
    grade: str = ""
    tier: str = "unknown"           # tier_1 | tier_2 | tier_3 | tier_4 | unknown


@dataclass(slots=True)
class RedrobSignals:
    """
    Redrob platform signals.

    Only the fields relevant to honeypot contradiction checks are typed
    explicitly.  The full behavioral multiplier is Member C's scope.
    """
    profile_completeness_score: float = 0.0
    verified_email: bool = False
    verified_phone: bool = False
    linkedin_connected: bool = False
    signup_date: str = ""
    last_active_date: str = ""


@dataclass(slots=True)
class CandidateRecord:
    """
    Fully typed representation of one candidate from candidates.jsonl.

    Maps 1:1 to the candidate_schema.json fields.  This is the canonical
    in-memory representation consumed by all pipeline stages.
    """
    # Profile-level fields
    candidate_id: str = ""
    years_of_experience: float = 0.0
    current_title: str = ""
    current_company: str = ""
    current_company_size: str = ""
    current_industry: str = ""
    headline: str = ""
    summary: str = ""
    location: str = ""

    # Nested collections
    career_history: list[CareerEntry] = field(default_factory=list)
    education: list[EducationEntry] = field(default_factory=list)
    skills: list[SkillEntry] = field(default_factory=list)

    # Platform signals
    redrob_signals: RedrobSignals = field(default_factory=RedrobSignals)

    # Optional arrays — may be absent in the data
    certifications: list[dict[str, Any]] = field(default_factory=list)
    languages: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_skill(raw: dict[str, Any]) -> SkillEntry:
    """Parse a single raw skill dict into a SkillEntry."""
    return SkillEntry(
        name=str(raw.get("name", "")).strip(),
        proficiency=str(raw.get("proficiency", "")).strip().lower(),
        endorsements=int(raw.get("endorsements", 0) or 0),
        duration_months=int(raw.get("duration_months", 0) or 0),
    )


def _parse_career(raw: dict[str, Any]) -> CareerEntry:
    """Parse a single raw career_history dict into a CareerEntry."""
    return CareerEntry(
        company=str(raw.get("company", "")).strip(),
        title=str(raw.get("title", "")).strip(),
        start_date=str(raw.get("start_date", "")).strip(),
        end_date=str(raw.get("end_date", "")).strip(),
        duration_months=int(raw.get("duration_months", 0) or 0),
        is_current=bool(raw.get("is_current", False)),
        industry=str(raw.get("industry", "")).strip(),
        company_size=str(raw.get("company_size", "")).strip(),
        description=str(raw.get("description", "")).strip(),
    )


def _parse_education(raw: dict[str, Any]) -> EducationEntry:
    """Parse a single raw education dict into an EducationEntry."""
    return EducationEntry(
        institution=str(raw.get("institution", "")).strip(),
        degree=str(raw.get("degree", "")).strip(),
        field_of_study=str(raw.get("field_of_study", "")).strip(),
        start_year=int(raw.get("start_year", 0) or 0),
        end_year=int(raw.get("end_year", 0) or 0),
        grade=str(raw.get("grade", "")).strip(),
        tier=str(raw.get("tier", "unknown")).strip().lower(),
    )


def _parse_redrob_signals(raw: Any) -> RedrobSignals:
    """
    Parse redrob_signals which may be a dict or a list of signal dicts.

    Handles both formats:
      - dict: {"profile_completeness_score": 85, "verified_email": true, ...}
      - list: [{"signal_name": "...", "value": ...}, ...]
    """
    if raw is None:
        return RedrobSignals()

    # If it's already a flat dict, use it directly
    if isinstance(raw, dict):
        return RedrobSignals(
            profile_completeness_score=float(
                raw.get("profile_completeness_score", 0) or 0
            ),
            verified_email=bool(raw.get("verified_email", False)),
            verified_phone=bool(raw.get("verified_phone", False)),
            linkedin_connected=bool(raw.get("linkedin_connected", False)),
            signup_date=str(raw.get("signup_date", "")).strip(),
            last_active_date=str(raw.get("last_active_date", "")).strip(),
        )

    # If it's a list of {signal_name, value} pairs, flatten
    if isinstance(raw, list):
        signals_dict: dict[str, Any] = {}
        for entry in raw:
            if isinstance(entry, dict) and "signal_name" in entry:
                signals_dict[entry["signal_name"]] = entry.get("value")
        return _parse_redrob_signals(signals_dict)

    return RedrobSignals()


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
    ValueError
        If candidate_id is missing or empty.
    """
    cid = str(raw.get("candidate_id", "")).strip()
    if not cid:
        raise ValueError(f"Missing or empty candidate_id in record: {raw!r:.200s}")

    # Parse nested profile fields — handle both flat and nested "profile" key
    profile = raw.get("profile", raw)

    return CandidateRecord(
        candidate_id=cid,
        years_of_experience=float(
            profile.get("years_of_experience", raw.get("years_of_experience", 0)) or 0
        ),
        current_title=str(
            profile.get("current_title", raw.get("current_title", ""))
        ).strip(),
        current_company=str(
            profile.get("current_company", raw.get("current_company", ""))
        ).strip(),
        current_company_size=str(
            profile.get("current_company_size", raw.get("current_company_size", ""))
        ).strip(),
        current_industry=str(
            profile.get("current_industry", raw.get("current_industry", ""))
        ).strip(),
        headline=str(
            profile.get("headline", raw.get("headline", ""))
        ).strip(),
        summary=str(
            profile.get("summary", raw.get("summary", ""))
        ).strip(),
        location=str(
            profile.get("location", raw.get("location", ""))
        ).strip(),
        career_history=[
            _parse_career(c) for c in (raw.get("career_history") or [])
        ],
        education=[
            _parse_education(e) for e in (raw.get("education") or [])
        ],
        skills=[
            _parse_skill(s) for s in (raw.get("skills") or [])
        ],
        redrob_signals=_parse_redrob_signals(raw.get("redrob_signals")),
        certifications=list(raw.get("certifications") or []),
        languages=list(raw.get("languages") or []),
    )


# ---------------------------------------------------------------------------
# Streaming loader
# ---------------------------------------------------------------------------

def load_candidates(path: str | Path) -> Iterator[CandidateRecord]:
    """
    Stream candidates from a JSONL file, yielding one parsed
    ``CandidateRecord`` per line.

    This is memory-efficient — it never loads the full 100k-record file
    into a list.  Parse errors on individual lines are logged and skipped.

    Parameters
    ----------
    path : str or Path
        Path to candidates.jsonl.

    Yields
    ------
    CandidateRecord
        One parsed candidate per JSONL line.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Candidates file not found: {path}")

    import sys

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                yield parse_candidate(raw)
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                print(
                    f"[WARN] Skipping line {line_num}: {exc}",
                    file=sys.stderr,
                )
