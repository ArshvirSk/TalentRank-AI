"""
Stage 5 (partial) — Behavioral scoring & availability multiplier.

Owner: Member 3

Derives the availability_multiplier (~0.6–1.15) from the 23 redrob_signals
present in each candidate record.  Also computes a normalized behavioral_score
for the composite formula.
"""

from __future__ import annotations

from src.features.schema import CandidateRecord

import config


def compute_availability_multiplier(candidate: CandidateRecord) -> float:
    """
    Compute the availability multiplier from redrob_signals.

    The multiplier adjusts the composite score based on the candidate's
    availability, notice period, willingness to relocate, and other
    behavioural signals.

    Parameters
    ----------
    candidate : CandidateRecord
        Typed candidate record with populated redrob_signals.

    Returns
    -------
    float
        Multiplier in the range
        [config.AVAILABILITY_MULTIPLIER_MIN, config.AVAILABILITY_MULTIPLIER_MAX].

    Raises
    ------
    NotImplementedError
        Scaffold — not yet implemented.

    Notes
    -----
    Implementation checklist (Member 3):
      1. Enumerate the 23 known redrob_signals (get list from Member 1's
         schema mapping).
      2. Define a contribution weight per signal type.
      3. Aggregate into a single multiplier, clamped to [0.6, 1.15].
      4. Handle missing signals gracefully (default to neutral = 1.0).
    """
    # TODO(Member 3): implement availability multiplier from redrob_signals
    raise NotImplementedError(
        "compute_availability_multiplier: not yet implemented"
    )


def compute_behavioral_score(candidate: CandidateRecord) -> float:
    """
    Compute a normalized behavioral score (0–1) from candidate signals.

    This feeds into the composite formula as the ``behavioral_score``
    component (weight = 0.20).

    Parameters
    ----------
    candidate : CandidateRecord
        Typed candidate record.

    Returns
    -------
    float
        Behavioral score in [0, 1].

    Raises
    ------
    NotImplementedError
        Scaffold — not yet implemented.

    Notes
    -----
    Implementation checklist (Member 3):
      1. Factor in notice period, willingness to relocate, job-seeking
         activity level, profile completeness.
      2. Normalize to [0, 1].
    """
    # TODO(Member 3): implement behavioral scoring
    raise NotImplementedError(
        "compute_behavioral_score: not yet implemented"
    )
