"""
Features sub-package — Data Engineering & Feature Pipeline.

Owner: Member 1

Stages covered:
  2 — Structured Feature Extraction
  6 — Honeypot Detection Layer
"""

from src.features.schema import parse_candidate, CandidateRecord
from src.features.career_signals import extract_career_signals, CareerSignals
from src.features.skill_trust import compute_skill_trust_score
from src.features.honeypot import detect_honeypot, HoneypotResult

__all__ = [
    "parse_candidate",
    "CandidateRecord",
    "extract_career_signals",
    "CareerSignals",
    "compute_skill_trust_score",
    "detect_honeypot",
    "HoneypotResult",
]
