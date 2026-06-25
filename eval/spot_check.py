"""
Evaluation — Manual face-validity spot checks across score deciles.

Owner: Member 4

Samples candidates from each score decile and presents them for manual
review to verify the pipeline isn't systematically misranking.
"""

from __future__ import annotations

from pathlib import Path

import config


def run_spot_check(
    submission_path: str | Path = config.SUBMISSION_PATH,
    candidates_path: str | Path = config.CANDIDATES_PATH,
    samples_per_decile: int = 3,
) -> None:
    """
    Sample candidates from each score decile for manual face-validity review.

    For each decile (top 10%, 10–20%, ..., 90–100%), print:
      - candidate_id, rank, score
      - key facts (skills, career highlights)
      - reasoning string
      - any honeypot flags

    Parameters
    ----------
    submission_path : str or Path
        Path to the submission CSV.
    candidates_path : str or Path
        Path to candidates.jsonl (for full record lookup).
    samples_per_decile : int
        How many candidates to sample from each decile.

    Raises
    ------
    NotImplementedError
        Scaffold — not yet implemented.
    """
    # TODO(Member 4): implement decile-based spot check sampling
    raise NotImplementedError(
        "run_spot_check: spot check sampling not yet implemented"
    )


if __name__ == "__main__":
    run_spot_check()
