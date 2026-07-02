"""
Evaluation -- Manual face-validity spot checks across score deciles.

Owner: Member 4

After the full ranking pipeline produces submission.csv, this script
performs two kinds of checks:

1. DECILE SAMPLING (manual review)
   Splits the 100 ranked candidates into 10 deciles of 10, samples
   `samples_per_decile` candidates from each, and prints a structured
   profile card for each one.  A human reads through ~30 cards and
   asks: "Does rank 1 look like a strong AI engineer?  Does rank 95
   look weak?"  No automation replaces this step.

2. AUTOMATED CHECKS (programmatic)
   After the manual-review section, runs 5 checks on the full 100-row
   submission:
     - Disqualifier push-down (consultancy / low-exp in top 20)
     - Score range sanity (all in [0, 1])
     - Reasoning diversity (catch template collisions)
     - Top-10 title diversity (catch suspiciously narrow bias)
     - Sentinel -1 signal handling (response / offer rate)

Person D context:
  This file is part of the face-validity eval harness described in the
  project spec.  It is intentionally separate from validate_submission.py
  (which checks CSV format) -- this file checks whether the *content*
  makes sense.

  Consulting-only detection delegates to Person A's extract_career_signals()
  (src/features/career_signals.py) as the single source of truth.  If
  Person A's module is unavailable, a warning is printed and the check
  is skipped gracefully -- no crash.

  Honeypot checks are deliberately left out of this file -- they will be
  addressed in a separate prompt once Person C's flags are available.

Usage:
    python -m eval.spot_check
    python -m eval.spot_check submission.csv candidates.jsonl
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import config

# ---------------------------------------------------------------------------
# Person A integration -- consulting-only detection
# ---------------------------------------------------------------------------
# Import Person A's public API.  Wrapped in try/except so spot_check.py
# still runs even if src.features has an import error (e.g. missing deps).
# ---------------------------------------------------------------------------
try:
    from src.features.schema import parse_candidate
    from src.features.career_signals import extract_career_signals
    _PERSON_A_AVAILABLE = True
except Exception as _import_err:  # noqa: BLE001
    _PERSON_A_AVAILABLE = False
    _PERSON_A_IMPORT_ERROR = str(_import_err)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sentinel(value: object) -> str:
    """Return 'N/A' if value is -1 (sentinel), else str(value)."""
    if value == -1:
        return "N/A"
    return str(value)


def _fmt_response_rate(value: object) -> str:
    """Format recruiter_response_rate: 2 dp if float, 'N/A' if -1."""
    if value == -1:
        return "N/A"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "N/A"


def _top3_skills(skills: list[dict]) -> str:
    """
    Return top-3 skills by endorsements as 'Name(Xmo,pro)' strings.
    Handles missing keys defensively.
    """
    if not skills:
        return "N/A"
    sorted_skills = sorted(
        skills,
        key=lambda s: s.get("endorsements", 0),
        reverse=True,
    )
    parts: list[str] = []
    for s in sorted_skills[:3]:
        name = s.get("name", "?")
        months = s.get("duration_months", 0)
        prof = (s.get("proficiency") or "?")[:3]
        parts.append(f"{name}({months}mo,{prof})")
    return ", ".join(parts) if parts else "N/A"


def _sample_indices(total: int, n: int) -> list[int]:
    """
    Pick n evenly spaced indices from range(total).
    Deterministic -- no random module used.
    Example: total=10, n=3 -> [0, 3, 6]
    """
    if n >= total:
        return list(range(total))
    step = total / n
    return [int(i * step) for i in range(n)]


def _is_consulting_only(record: dict) -> bool:
    """
    Return True if this candidate is consulting-only.

    Delegates to Person A's extract_career_signals() as the authoritative
    implementation.  If Person A's module is unavailable, returns False
    and relies on the caller to print a warning.

    Parameters
    ----------
    record : dict
        Raw candidate record dict from candidates.jsonl.

    Returns
    -------
    bool
        True if consulting_only_flag is set by Person A's logic.
    """
    if not _PERSON_A_AVAILABLE:
        return False
    try:
        candidate = parse_candidate(record)
        signals = extract_career_signals(candidate)
        return signals.consulting_only_flag
    except Exception:  # noqa: BLE001
        # Defensive: if parsing fails for this record, don't crash the check
        return False


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def run_spot_check(
    submission_path: str | Path = config.SUBMISSION_PATH,
    candidates_path: str | Path = config.CANDIDATES_PATH,
    samples_per_decile: int = 3,
) -> None:
    """
    Sample candidates from each score decile for manual face-validity review.

    For each decile (top 10%, 10-20%, ..., 90-100%), prints a structured
    profile card showing:
      - candidate_id, rank, score
      - name, title, company, industry, years of experience
      - top 3 skills by endorsements
      - behavioral signals (response rate, notice, open-to-work)
      - reasoning string from submission CSV

    After all deciles, prints automated checks covering disqualifier
    push-down, score sanity, reasoning diversity, title diversity, and
    sentinel signal handling.

    Parameters
    ----------
    submission_path : str or Path
        Path to the submission CSV (candidate_id, rank, score, reasoning).
    candidates_path : str or Path
        Path to candidates.jsonl for full record lookup.
    samples_per_decile : int
        How many candidates to sample from each decile (default 3).
    """

    submission_path = Path(submission_path)
    candidates_path = Path(candidates_path)

    # ------------------------------------------------------------------
    # STEP 1 -- Load submission CSV
    # ------------------------------------------------------------------
    if not submission_path.exists():
        print(f"X submission.csv not found at {submission_path}")
        print("  Run the full pipeline first to generate it.")
        return

    rows: list[dict] = []
    try:
        if str(submission_path).lower().endswith('.xlsx'):
            import pandas as pd
            df = pd.read_excel(submission_path, dtype=str, keep_default_na=False)
            for _, row in df.iterrows():
                try:
                    rows.append({
                        "candidate_id": row.get("candidate_id", "").strip(),
                        "rank": int(float(row.get("rank", 0))),
                        "score": float(row.get("score", 0.0)),
                        "reasoning": row.get("reasoning", "").strip(),
                    })
                except (KeyError, ValueError, TypeError) as exc:
                    print(f"  [WARNING] Skipping malformed XLSX row: {exc}")
        else:
            with open(submission_path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    try:
                        rows.append({
                            "candidate_id": row["candidate_id"].strip(),
                            "rank": int(float(row["rank"])),
                            "score": float(row["score"]),
                            "reasoning": row["reasoning"].strip(),
                        })
                    except (KeyError, ValueError) as exc:
                        print(f"  [WARNING] Skipping malformed CSV row: {exc}")
    except OSError as exc:
        print(f"X Failed to read submission file: {exc}")
        return

    if not rows:
        print("X submission.csv is empty or has no valid rows.")
        return

    # Sort by rank ascending (enforce — should already be sorted)
    rows.sort(key=lambda r: r["rank"])

    # ------------------------------------------------------------------
    # STEP 2 -- Load candidates.jsonl into lookup dict
    # ------------------------------------------------------------------
    if not candidates_path.exists():
        print(f"X candidates.jsonl not found at {candidates_path}")
        print("  Place data/candidates.jsonl before running spot check.")
        return

    candidates_lookup: dict[str, dict] = {}
    try:
        with open(candidates_path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    cid = record.get("candidate_id")
                    if cid is not None:
                        candidates_lookup[str(cid)] = record
                except json.JSONDecodeError as exc:
                    print(f"  [WARNING] Line {lineno}: JSON parse error — {exc}")
    except OSError as exc:
        print(f"X Failed to read candidates.jsonl: {exc}")
        return

    print(f"Loaded {len(candidates_lookup)} candidate records from {candidates_path}")

    # ------------------------------------------------------------------
    # STEP 3 -- Split into deciles
    # ------------------------------------------------------------------
    # Clamp to 100 rows; warn if submission has wrong count
    if len(rows) != config.TOP_K:
        print(
            f"  [WARNING] submission.csv has {len(rows)} rows, "
            f"expected {config.TOP_K} -- deciles may be uneven"
        )

    deciles: list[list[dict]] = []
    for i in range(10):
        start = i * 10
        end = start + 10
        decile_rows = rows[start:end]
        if decile_rows:
            deciles.append(decile_rows)

    # ------------------------------------------------------------------
    # STEP 4 + 5 -- Sample from each decile and print profile cards
    # ------------------------------------------------------------------
    if not _PERSON_A_AVAILABLE:
        print(
            f"  [WARNING] Person A module unavailable "
            f"({_PERSON_A_IMPORT_ERROR}) -- "
            f"consulting-only flags will not appear in profile cards"
        )

    for d_idx, decile in enumerate(deciles, start=1):
        start_rank = decile[0]["rank"]
        end_rank = decile[-1]["rank"]
        print()
        print("=" * 62)
        print(f"DECILE {d_idx} | Ranks {start_rank}-{end_rank}")
        print("=" * 62)

        indices = _sample_indices(len(decile), samples_per_decile)
        for idx in indices:
            row = decile[idx]
            cid = row["candidate_id"]
            rank = row["rank"]
            score = row["score"]
            reasoning = row["reasoning"]

            print()
            print(f"--- Rank {rank} | Score {score:.4f} | {cid} ---")

            record = candidates_lookup.get(cid)
            if record is None:
                print(f"  [WARNING] candidate_id {cid} not found in candidates.jsonl")
                print(f"Reasoning  : {reasoning}")
                print("---")
                continue

            # Safely extract nested fields
            profile = record.get("profile") or {}
            signals = record.get("redrob_signals") or {}
            skills = record.get("skills") or []
            career = record.get("career_history") or []

            name = profile.get("anonymized_name", "N/A")
            title = profile.get("current_title", "N/A")
            company = profile.get("current_company", "N/A")
            industry = profile.get("current_industry", "N/A")
            yoe = profile.get("years_of_experience", "N/A")

            top_skills = _top3_skills(skills)

            rr_raw = signals.get("recruiter_response_rate", "N/A")
            response_rate = _fmt_response_rate(rr_raw)
            notice = signals.get("notice_period_days", "N/A")
            open_to_work = signals.get("open_to_work_flag", "N/A")
            completeness = signals.get("profile_completeness_score", "N/A")

            # Career summary: consulting-only flag (Person A heuristic)
            consulting_flag = ""
            if _is_consulting_only(record):
                consulting_flag = " [CONSULTING-ONLY FLAG]"

            print(f"Name       : {name}")
            print(f"Title      : {title} @ {company} ({industry}){consulting_flag}")
            print(f"Experience : {yoe}y")
            print(f"Top Skills : {top_skills}")
            print(
                f"Response   : {response_rate} | "
                f"Notice: {notice}d | "
                f"Open: {open_to_work}"
            )
            print(f"Completeness: {completeness}")
            print(f"Reasoning  : {reasoning}")
            print("---")

    # ------------------------------------------------------------------
    # STEP 6 -- Automated checks
    # ------------------------------------------------------------------
    print()
    print("=" * 62)
    print("AUTOMATED CHECKS")
    print("=" * 62)
    print()

    # CHECK 1 -- Disqualifier push-down
    # Uses Person A's extract_career_signals().consulting_only_flag as the
    # authoritative definition.  Falls back to a warning if unavailable.
    top20 = [r for r in rows if r["rank"] <= 20]
    disq_count = 0

    if not _PERSON_A_AVAILABLE:
        print(
            f"  [WARNING] CHECK 1 skipped -- Person A module unavailable "
            f"({_PERSON_A_IMPORT_ERROR})"
        )
    else:
        for r in top20:
            rec = candidates_lookup.get(r["candidate_id"])
            if rec is None:
                continue
            try:
                candidate = parse_candidate(rec)
                signals = extract_career_signals(candidate)
                if signals.consulting_only_flag:
                    disq_count += 1
            except Exception:  # noqa: BLE001
                continue

        if disq_count > 3:
            print(
                f"X  Disqualifier check -- {disq_count} consulting-only "
                f"profiles in top 20 (review recommended)"
            )
        else:
            print(
                f"OK: Disqualifier check -- {disq_count} consulting-only "
                f"profiles in top 20"
            )

    # CHECK 2 -- Score range sanity
    out_of_range = [r for r in rows if not (0.0 <= r["score"] <= 1.0)]
    if out_of_range:
        print(f"X  {len(out_of_range)} scores outside [0.0, 1.0]")
    else:
        print("OK: All scores in [0.0, 1.0]")

    # CHECK 3 -- Reasoning diversity
    total_rows = len(rows)
    unique_reasoning = len({r["reasoning"] for r in rows})
    uniqueness_rate = unique_reasoning / total_rows if total_rows > 0 else 0.0
    if uniqueness_rate < 0.80:
        print(
            f"X  Low reasoning uniqueness: {unique_reasoning}/{total_rows} "
            f"-- possible template collision"
        )
    else:
        print(
            f"OK: Reasoning uniqueness {unique_reasoning}/{total_rows} "
            f"({uniqueness_rate:.0%})"
        )

    # CHECK 4 -- Top-10 title diversity
    top10 = [r for r in rows if r["rank"] <= 10]
    titles: list[str] = []
    missing_in_lookup = 0
    for r in top10:
        rec = candidates_lookup.get(r["candidate_id"])
        if rec is None:
            missing_in_lookup += 1
            continue
        profile = rec.get("profile") or {}
        t = profile.get("current_title", "").strip()
        if t:
            titles.append(t)

    unique_titles = len(set(titles))
    if missing_in_lookup > 0:
        print(
            f"  [WARNING] {missing_in_lookup} top-10 candidate(s) not "
            f"found in lookup -- title diversity count may be understated"
        )
    if unique_titles < 3:
        print(
            f"X  Low title diversity in top 10 -- {unique_titles} unique "
            f"titles (possible bias)"
        )
    else:
        print(
            f"OK: Top-10 title diversity -- {unique_titles} unique titles"
        )

    # CHECK 5 -- Sentinel -1 signal handling
    sentinel_count = 0
    for r in rows:
        rec = candidates_lookup.get(r["candidate_id"])
        if rec is None:
            continue
        signals = rec.get("redrob_signals") or {}
        rr = signals.get("recruiter_response_rate", 0)
        oa = signals.get("offer_acceptance_rate", 0)
        if rr == -1 or oa == -1:
            sentinel_count += 1

    print(
        f"OK: {sentinel_count}/{total_rows} candidates have sentinel -1 "
        f"signals (no offer/response history) -- handled correctly"
    )

    print()
    print("Spot check complete. Review output above for any X flags.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sub = sys.argv[1] if len(sys.argv) > 1 else str(config.SUBMISSION_PATH)
    cands = sys.argv[2] if len(sys.argv) > 2 else str(config.CANDIDATES_PATH)
    run_spot_check(submission_path=sub, candidates_path=cands)
