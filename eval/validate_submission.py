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
import re
import sys
from pathlib import Path

import config

# Candidate ID format required by the judges' validator
_CANDIDATE_ID_RE = re.compile(r"^CAND_[0-9]{7}$")


def validate_submission(
    submission_path: str | Path = config.SUBMISSION_PATH,
    candidates_path: str | Path | None = None,
    strict: bool = True,
) -> bool:
    """
    Validate a submission CSV against hackathon format requirements.

    Always collects all errors before reporting — never stops at the first
    failure.  The ``strict`` parameter is accepted for API compatibility
    but has no effect on behavior.

    Parameters
    ----------
    submission_path : str or Path
        Path to the submission CSV.
    candidates_path : str or Path, optional
        Path to candidates.jsonl for candidate_id cross-checking.
        If None, candidate_id existence is not verified.
    strict : bool
        Accepted for API compatibility.  Ignored — all errors are always
        collected and reported regardless of this value.

    Returns
    -------
    bool
        True if the submission passes all checks, False otherwise.

    Checks performed
    ----------------
    1. Header row matches exactly: candidate_id, rank, score, reasoning
    2. Exactly config.TOP_K (100) data rows present
    3. Each candidate_id matches format CAND_XXXXXXX (7 digits)
    4. Ranks are sequential 1..100 with no gaps or duplicates
    5. Scores are floats in [0.0, 1.0]
    6. Reasoning is non-empty for every row
    7. Scores are non-increasing by rank (rank 1 >= rank 2 >= ...)
    8. Tie-breaking: equal scores must have candidate_id ascending
    9. (Optional) All candidate_ids exist in candidates.jsonl
    """
    import json

    _ = strict  # accepted for API compatibility; all errors are always collected
    errors: list[str] = []
    expected_header = ["candidate_id", "rank", "score", "reasoning"]
    submission_path = Path(submission_path)

    try:
        if str(submission_path).lower().endswith('.xlsx'):
            import pandas as pd
            df = pd.read_excel(submission_path, dtype=str, keep_default_na=False)
            if df.empty and df.columns.empty:
                errors.append("File is empty")
                rows_raw: list[list[str]] = []
                header = None
            else:
                header = df.columns.tolist()
                rows_raw = df.values.tolist()
        else:
            with open(submission_path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                except StopIteration:
                    errors.append("CSV file is empty")
                    rows_raw: list[list[str]] = []
                    header = None
                else:
                    rows_raw = list(reader)
    except FileNotFoundError:
        print(f"X File not found: {submission_path}")
        return False
    except OSError as exc:
        print(f"X Failed to read CSV: {exc}")
        return False

    if header is not None:
        if header != expected_header:
            errors.append(
                "Header mismatch. Expected: candidate_id,rank,score,reasoning. "
                f"Got: {','.join(header)}"
            )

        if len(rows_raw) != config.TOP_K:
            errors.append(f"Row count is {len(rows_raw)}, expected {config.TOP_K}")

        ranks_seen: list[int] = []
        candidate_ids: list[str] = []

        for i, row in enumerate(rows_raw, start=1):
            if len(row) != 4:
                errors.append(f"Row {i}: expected 4 columns, got {len(row)}")
                continue

            cid, rank_str, score_str, reasoning = row

            if not str(cid).strip():
                errors.append(f"Row {i}: empty candidate_id")
            elif not _CANDIDATE_ID_RE.match(str(cid).strip()):
                errors.append(
                    f"Row {i}: candidate_id '{cid}' does not match "
                    f"required format CAND_XXXXXXX (7 digits)"
                )
            else:
                candidate_ids.append(str(cid).strip())

            try:
                rank_val = float(rank_str)
                if not rank_val.is_integer():
                    errors.append(f"Row {i}: rank must be an integer, got {rank_str!r}")
                else:
                    ranks_seen.append(int(rank_val))
            except (ValueError, TypeError):
                errors.append(f"Row {i}: rank is not parseable as a number: {rank_str!r}")

            try:
                score_val = float(score_str)
                if not 0.0 <= score_val <= 1.0:
                    errors.append(f"Row {i}: score {score_val} outside [0.0, 1.0]")
            except (ValueError, TypeError):
                errors.append(f"Row {i}: score is not parseable as a float: {score_str!r}")

            if not str(reasoning).strip():
                errors.append(f"Row {i}: empty reasoning")

        expected_ranks = set(range(1, config.TOP_K + 1))
        actual_ranks = set(ranks_seen)
        if actual_ranks != expected_ranks or len(ranks_seen) != len(actual_ranks):
            errors.append(
                f"Ranks must be exactly {{1, 2, ..., {config.TOP_K}}} "
                "with no duplicates or gaps"
            )

        if len(candidate_ids) != len(set(candidate_ids)):
            seen: set[str] = set()
            dupes: list[str] = []
            for cid in candidate_ids:
                if cid in seen and cid not in dupes:
                    dupes.append(cid)
                seen.add(cid)
            errors.append(f"Duplicate candidate_ids found: {dupes[:5]}")

        # Build (rank, score, candidate_id) triples for ordering checks
        by_rank: list[tuple[int, float, str]] = []
        for row in rows_raw:
            if len(row) != 4:
                continue
            cid_r, rank_str_r, score_str_r, _ = row
            try:
                rank_r = int(float(rank_str_r))
                score_r = float(score_str_r)
                by_rank.append((rank_r, score_r, str(cid_r).strip()))
            except (ValueError, TypeError):
                pass
        by_rank.sort(key=lambda x: x[0])

        # Check 7: scores must be non-increasing by rank
        for j in range(len(by_rank) - 1):
            r1, s1, _ = by_rank[j]
            r2, s2, _ = by_rank[j + 1]
            if s1 < s2:
                errors.append(
                    f"Score must be non-increasing by rank: "
                    f"rank {r1} ({s1}) < rank {r2} ({s2})"
                )
                break  # report first violation only to avoid flooding

        # Check 8: tie-breaking — equal scores must have candidate_id ascending
        for j in range(len(by_rank) - 1):
            r1, s1, c1 = by_rank[j]
            r2, s2, c2 = by_rank[j + 1]
            if s1 == s2 and c1 > c2:
                errors.append(
                    f"Tie-break violation at ranks {r1}/{r2}: "
                    f"equal scores require candidate_id ascending "
                    f"('{c1}' > '{c2}')"
                )

        # Check 9: reasoning uniqueness — catch template collisions
        valid_rows = [row for row in rows_raw if len(row) == 4]
        if valid_rows:
            unique_count = len({row[3] for row in valid_rows})
            uniqueness_rate = unique_count / len(valid_rows)
            if uniqueness_rate < 0.80:
                errors.append(
                    f"Low reasoning uniqueness: {unique_count}/{len(valid_rows)} "
                    f"unique strings ({uniqueness_rate:.0%}) -- possible template collision"
                )

        if candidates_path is not None:
            valid_ids: set[str] = set()
            with open(candidates_path, encoding="utf-8") as cf:
                for line in cf:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    cid = record.get("candidate_id")
                    if cid is not None:
                        valid_ids.add(str(cid))

            missing = [cid for cid in candidate_ids if cid not in valid_ids]
            if missing:
                errors.append(
                    f"{len(missing)} candidate_id(s) not found in candidates.jsonl. "
                    f"Examples: {missing[:5]}"
                )

    if errors:
        print(f"VALIDATION FAILED - {len(errors)} error(s):")
        for error in errors:
            print(f"  X {error}")
        return False

    print(f"OK: Submission valid - {config.TOP_K} candidates")
    return True


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else str(config.SUBMISSION_PATH)
    result = validate_submission(path)
    sys.exit(0 if result else 1)
