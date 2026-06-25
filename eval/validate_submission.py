"""
Evaluation — Submission CSV validator.

Owner: Member 4

Validates that the submission CSV meets the hackathon's format requirements:
  - Exactly 100 rows (excluding header)
  - Columns: candidate_id, rank, score, reasoning
  - Ranks are 1–100 with no gaps or duplicates
  - All candidate_ids are present in candidates.jsonl
  - Scores are numeric and in a reasonable range
  - Reasoning is non-empty for every row
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import config


def validate_submission(
    submission_path: str | Path = config.SUBMISSION_PATH,
    candidates_path: str | Path | None = None,
    strict: bool = True,
) -> bool:
    """
    Validate a submission CSV against hackathon format requirements.

    Parameters
    ----------
    submission_path : str or Path
        Path to the submission CSV.
    candidates_path : str or Path, optional
        Path to candidates.jsonl for candidate_id cross-checking.
        If None, candidate_id existence is not verified.
    strict : bool
        If True, raise on first error.  If False, collect all errors and
        print a summary.

    Returns
    -------
    bool
        True if the submission passes all checks.

    Raises
    ------
    NotImplementedError
        Scaffold — not yet implemented.

    Notes
    -----
    Implementation checklist (Member 4):
      1. Read CSV, verify header row matches expected columns.
      2. Count rows (must be exactly config.TOP_K = 100).
      3. Verify ranks are sequential 1..100.
      4. Verify scores are float-parseable and non-negative.
      5. Verify reasoning is non-empty for every row.
      6. (Optional) Cross-check candidate_ids against candidates.jsonl.
      7. Print PASS/FAIL summary.
    """
    # TODO(Member 4): implement submission validation
    raise NotImplementedError(
        "validate_submission: validation logic not yet implemented"
    )


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else str(config.SUBMISSION_PATH)
    result = validate_submission(path)
    sys.exit(0 if result else 1)
