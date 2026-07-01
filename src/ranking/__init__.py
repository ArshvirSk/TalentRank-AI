"""
Ranking sub-package — Ranking Engine & ML.

Owner: Member 3

Stages covered:
  5 — Hybrid Candidate Scoring
  7 — Final Ranking Engine (rank.py CLI)

Note: rank.py is imported lazily to avoid circular/heavy imports at startup.
"""

from src.ranking.fusion import compute_composite_score, ScoreBreakdown
from src.ranking.behavioral import compute_availability_multiplier

__all__ = [
    "compute_composite_score",
    "ScoreBreakdown",
    "compute_availability_multiplier",
]
