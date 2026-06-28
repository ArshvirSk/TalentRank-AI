"""
Stage 8 -- Explanation Generator.

Owner: Member 4

Generates compact, numeric, semicolon-separated reasoning strings for the
submission CSV.  Slot-fills templates from extracted facts only -- zero
hallucination risk.

Key constraints:
  - Every value must come from breakdown fields or candidate_facts.
  - Code must never crash if candidate_facts is None or empty.
  - No random, no pandas, no new top-level imports.
  - Template selection is deterministic: hash(candidate_id + str(rank)) % n.
  - jd_profile is accepted but currently unused (assigned to _).
"""

from __future__ import annotations

from src.ranking.fusion import ScoreBreakdown
from src.retrieval.jd_parser import JDProfile


def get_top_gap_reason(breakdown: ScoreBreakdown, facts: dict) -> str:
    """
    Return the most salient gap reason for a borderline candidate.

    Reads boolean flags from ``facts`` first (populated by Members A / C).
    Falls back to numeric thresholds on ``breakdown`` when flags are absent.
    Uses only ``.get()`` — never crashes on missing keys or fields.

    Parameters
    ----------
    breakdown : ScoreBreakdown
        Full score breakdown (always present, typed fields).
    facts : dict
        Optional enrichment dict from upstream members.  May be empty.

    Returns
    -------
    str
        A short, human-readable gap reason string.
    """
    # Specific flags from Person A / C — safe .get() with False defaults
    # Accepts both Person D's original key names and Person A's actual key names
    if facts.get("honeypot_flag", False):
        return "profile consistency concerns"
    if facts.get("langchain_only_flag", False) or facts.get("recent_llm_only_flag", False):
        return "limited ML depth beyond recent LLM tooling"
    if facts.get("short_tenure_flag", False) or facts.get("title_chaser_score", 0) > 0.5:
        return "limited production deployment evidence"
    if facts.get("consulting_only_flag", False):
        return "services/consulting background without product experience"

    # Numeric fallbacks — always available on breakdown
    if breakdown.seniority_and_shipping < 0.30:
        return "low production shipping signal"
    if breakdown.skill_match < 0.30:
        return "weak must-have skill coverage"
    if breakdown.career_match < 0.30:
        return "limited career alignment with JD"

    # Generic fallback — guaranteed non-empty
    return "partial alignment on must-have criteria"


def generate_reasoning(
    breakdown: ScoreBreakdown,
    jd_profile: JDProfile,
    rank: int,
    candidate_facts: dict | None = None,
) -> str:
    """
    Generate a compact reasoning string for one ranked candidate.

    Parameters
    ----------
    breakdown : ScoreBreakdown
        Full score breakdown from Stage 5.  All 8 numeric fields are used.
    jd_profile : JDProfile
        Parsed JD profile.  Accepted for API compatibility; currently unused.
    rank : int
        Candidate's rank (1-indexed).
    candidate_facts : dict, optional
        Enrichment dict from Members A / C.  Every key is optional -- the
        function degrades gracefully to score-only clauses when absent.

    Returns
    -------
    str
        Single-line, semicolon-separated reasoning string suitable for the
        ``reasoning`` column of submission.csv.
    """
    # Imports inside function body to avoid circular imports at module load
    from src.explain.templates import (
        BEHAVIORAL_SCORES_ONLY_TEMPLATES,
        BEHAVIORAL_TEMPLATES,
        CAREER_SCORES_ONLY_TEMPLATES,
        CAREER_TEMPLATES,
        DISQUALIFIER_TEMPLATES,
        GAP_ACKNOWLEDGMENT_TEMPLATES,
        HONEYPOT_TEMPLATES,
        NO_PENALTY_TEMPLATES,
        OPENING_TEMPLATES,
        SKILL_CLAUSE_SCORES_ONLY,
        SKILL_CLAUSE_WITH_DATA,
        get_template_variant,
    )

    # ------------------------------------------------------------------
    # Step 0 — Setup
    # ------------------------------------------------------------------
    _ = jd_profile  # accepted, not yet used
    facts: dict = candidate_facts or {}
    cid: str = breakdown.candidate_id or facts.get("candidate_id", f"rank_{rank}")

    # ------------------------------------------------------------------
    # Step 1 — Build SKILL CLAUSE
    # ------------------------------------------------------------------
    top_skills: list[str] = facts.get("top_skills") or []
    top_skill_months: list[int] = facts.get("top_skill_months") or []
    match_count = facts.get("match_count")
    total_count = facts.get("total_count")

    if top_skills:
        # Build "MLOps 27mo, PyTorch 19mo, K8s 13mo" — use "?" when months absent
        pairs: list[str] = []
        for i, skill in enumerate(top_skills[:3]):
            months = top_skill_months[i] if i < len(top_skill_months) else "?"
            pairs.append(f"{skill} {months}mo")
        skills_with_months = ", ".join(pairs)

        mc = match_count if match_count is not None else "?"
        tc = total_count if total_count is not None else "?"

        template = get_template_variant(SKILL_CLAUSE_WITH_DATA, cid, rank)
        skill_clause = template.format(
            match_count=mc,
            total_count=tc,
            skills_with_months=skills_with_months,
            skill_match=breakdown.skill_match,
        )
    else:
        template = get_template_variant(SKILL_CLAUSE_SCORES_ONLY, cid, rank)
        skill_clause = template.format(
            skill_match=breakdown.skill_match,
            career_match=breakdown.career_match,
        )

    # ------------------------------------------------------------------
    # Step 2 — Build OPENING segment
    # ------------------------------------------------------------------
    template = get_template_variant(OPENING_TEMPLATES, cid, rank)
    opening = template.format(
        rank=rank,
        score=breakdown.final_score,
        skill_clause=skill_clause,
    )

    # ------------------------------------------------------------------
    # Step 2b -- Optional JD-aware segment
    # Fires only if jd_profile is provided AND has must_have_skills.
    # Completely silent if jd_profile is None -- no change to existing output.
    # ------------------------------------------------------------------
    jd_seg = ""
    try:
        if (
            jd_profile is not None
            and hasattr(jd_profile, "must_have_skills")
            and jd_profile.must_have_skills
            and top_skills
        ):
            jd_must_haves = {s.lower().strip() for s in jd_profile.must_have_skills}
            candidate_top = [s.lower().strip() for s in top_skills[:5]]
            matched_jd = [s for s in candidate_top if s in jd_must_haves]
            missing_jd = [s for s in jd_must_haves if s not in candidate_top]
            parts_jd: list[str] = []
            if matched_jd:
                parts_jd.append(f"JD match: {', '.join(matched_jd[:3])}")
            if missing_jd:
                parts_jd.append(f"JD gap: {', '.join(list(missing_jd)[:2])}")
            if parts_jd:
                jd_seg = "; ".join(parts_jd)
    except Exception:  # noqa: BLE001
        # Never let JD parsing crash the generator
        jd_seg = ""

    # ------------------------------------------------------------------
    # Step 3 — Build CAREER segment
    # ------------------------------------------------------------------
    years = facts.get("years_of_experience")
    roles = facts.get("total_roles")
    current_role_months = facts.get("current_role_months")
    production_ml_months = facts.get("production_ml_months")
    domains: list[str] = facts.get("domains") or []

    if years is not None or roles is not None or current_role_months is not None:
        template = get_template_variant(CAREER_TEMPLATES, cid, rank)
        career_seg = template.format(
            years=years if years is not None else "?",
            roles=roles if roles is not None else "?",
            current_role_months=current_role_months if current_role_months is not None else "?",
        )
        # Optional production ML suffix
        if production_ml_months is not None:
            career_seg += f"; prod ML {production_ml_months}mo"
        # Optional domains suffix
        if domains:
            career_seg += f"; domains: {', '.join(domains[:3])}"
    else:
        template = get_template_variant(CAREER_SCORES_ONLY_TEMPLATES, cid, rank)
        career_seg = template.format(
            career_match=breakdown.career_match,
            seniority=breakdown.seniority_and_shipping,
        )
        if production_ml_months is not None:
            career_seg += f"; prod ML {production_ml_months}mo"
        if domains:
            career_seg += f"; domains: {', '.join(domains[:3])}"

    # ------------------------------------------------------------------
    # Step 4 — Build BEHAVIORAL segment
    # ------------------------------------------------------------------
    response_rate_raw = facts.get("response_rate")
    notice_period = facts.get("notice_period_days")
    open_to_work = facts.get("open_to_work")

    if open_to_work is True:
        work_status = "open to work"
    elif open_to_work is False:
        work_status = "not actively looking"
    else:
        work_status = "status unknown"

    # Use "?" string for missing response_rate to avoid misleading 0.0
    response_rate_display: str
    if response_rate_raw is not None:
        response_rate_display = f"{response_rate_raw:.2f}"
    else:
        response_rate_display = "?"

    notice_period_display = notice_period if notice_period is not None else "?"

    if response_rate_raw is not None or notice_period is not None:
        template = get_template_variant(BEHAVIORAL_TEMPLATES, cid, rank)
        behavioral_seg = template.format(
            response_rate=response_rate_display,
            notice_period=notice_period_display,
            work_status=work_status,
            avail_mult=breakdown.availability_multiplier,
            behavioral=breakdown.behavioral_score,
            avail_stability=breakdown.availability_stability,
        )
    else:
        template = get_template_variant(BEHAVIORAL_SCORES_ONLY_TEMPLATES, cid, rank)
        behavioral_seg = template.format(
            behavioral=breakdown.behavioral_score,
            avail_mult=breakdown.availability_multiplier,
            avail_stability=breakdown.availability_stability,
        )

    # ------------------------------------------------------------------
    # Step 5 — Build DISQUALIFIER / HONEYPOT segment
    # ------------------------------------------------------------------
    disqualifier_flag: bool = facts.get("disqualifier_flag", False)
    disqualifier_areas: list[str] = facts.get("disqualifier_areas") or []
    honeypot_flag: bool = facts.get("honeypot_flag", False)
    penalty: float = breakdown.disqualifier_penalty

    flag_parts: list[str] = []

    if disqualifier_flag and disqualifier_areas:
        template = get_template_variant(DISQUALIFIER_TEMPLATES, cid, rank)
        flag_parts.append(template.format(
            areas=", ".join(disqualifier_areas),
            penalty=penalty,
        ))
    elif penalty > 0:
        # Penalty present but no areas provided — still surface it
        flag_parts.append(f"disqualifier penalty {penalty:.2f} applied")

    if honeypot_flag:
        flag_parts.append(get_template_variant(HONEYPOT_TEMPLATES, cid, rank))

    if not flag_parts:
        flag_parts.append(get_template_variant(NO_PENALTY_TEMPLATES, cid, rank))

    flag_seg = "; ".join(flag_parts)

    # ------------------------------------------------------------------
    # Step 5b — Gap acknowledgment for borderline ranks
    # Fires when rank >= config.BORDERLINE_RANK_THRESHOLD or
    # final_score < config.BORDERLINE_SCORE_THRESHOLD.
    # Empty string for top candidates — no-op in assembly.
    # ------------------------------------------------------------------
    import config as _config
    if rank >= _config.BORDERLINE_RANK_THRESHOLD or breakdown.final_score < _config.BORDERLINE_SCORE_THRESHOLD:
        top_gap = get_top_gap_reason(breakdown, facts)
        gap_template = get_template_variant(GAP_ACKNOWLEDGMENT_TEMPLATES, cid, rank)
        gap_seg = gap_template.format(
            rank=rank,
            skill_match=breakdown.skill_match,
            top_gap=top_gap,
        )
    else:
        gap_seg = ""

    # ------------------------------------------------------------------
    # Step 6 — Assemble
    # ------------------------------------------------------------------
    parts = [opening, jd_seg, career_seg, behavioral_seg, flag_seg, gap_seg]
    result = "; ".join(p.strip() for p in parts if p and p.strip())

    if not result or not result.strip():
        result = f"Rank {rank}: composite score {breakdown.final_score:.3f}"

    return result


# ----------------------------------------------------------------------
# Smoke test — run via: python -m src.explain.generate
# Uses ScoreBreakdown dataclass directly; jd_profile passed as None.
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import os
    from pathlib import Path

    from src.ranking.fusion import ScoreBreakdown
    from src.retrieval.jd_parser import JDProfile

    # ------------------------------------------------------------------
    # Try to load a real candidate from sample_candidates.json fixture
    # ------------------------------------------------------------------
    fixture_path = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "sample_candidates.json"
    real_candidate: dict | None = None

    if fixture_path.exists():
        with open(fixture_path, encoding="utf-8") as fh:
            candidates = json.load(fh)
        if candidates:
            raw = candidates[0]  # TEST_001 — Senior ML Engineer
            # Build candidate_facts from the fixture schema
            skills = raw.get("skills", [])
            top_skills = [s["name"] for s in skills[:3]]
            top_skill_months = [s.get("duration_months", 0) for s in skills[:3]]
            career = raw.get("career_history", [])
            companies = [c["company"] for c in career]
            current_role = career[0] if career else {}
            signals = {s["signal_name"]: s["value"] for s in raw.get("redrob_signals", [])}
            real_candidate = {
                "candidate_id": raw["candidate_id"],
                "top_skills": top_skills,
                "top_skill_months": top_skill_months,
                "match_count": 7,        # mock — Member B supplies this
                "total_count": 10,       # mock — Member B supplies this
                "years_of_experience": raw.get("years_of_experience"),
                "total_roles": len(career),
                "current_role_months": current_role.get("duration_months"),
                "companies": companies,
                "domains": ["NLP", "RecSys"],
                "production_ml_months": 42,
                "response_rate": 0.71,
                "notice_period_days": signals.get("notice_period_days"),
                "open_to_work": signals.get("actively_looking", False),
                "disqualifier_flag": False,
                "disqualifier_areas": [],
                "honeypot_flag": False,
            }

    # ------------------------------------------------------------------
    # Mock ScoreBreakdowns for smoke test
    # ------------------------------------------------------------------
    mock_breakdown_1 = ScoreBreakdown(
        candidate_id="CAND_0000001",
        skill_match=0.91,
        career_match=0.78,
        behavioral_score=0.82,
        availability_stability=0.74,
        seniority_and_shipping=0.85,
        disqualifier_penalty=0.0,
        availability_multiplier=1.08,
        base_score=0.855,
        final_score=0.923,
    )
    mock_breakdown_2 = ScoreBreakdown(
        candidate_id="CAND_0004989",
        skill_match=0.61,
        career_match=0.55,
        behavioral_score=0.70,
        availability_stability=0.60,
        seniority_and_shipping=0.50,
        disqualifier_penalty=0.15,
        availability_multiplier=0.85,
        base_score=0.712,
        final_score=0.605,
    )
    mock_breakdown_3 = ScoreBreakdown(
        candidate_id="CAND_0099999",
        skill_match=0.22,
        career_match=0.18,
        behavioral_score=0.40,
        availability_stability=0.35,
        seniority_and_shipping=0.20,
        disqualifier_penalty=0.0,
        availability_multiplier=0.65,
        base_score=0.28,
        final_score=0.182,
    )

    facts_1 = {
        "top_skills": ["MLOps", "PyTorch", "Kubernetes"],
        "top_skill_months": [27, 19, 13],
        "match_count": 9,
        "total_count": 10,
        "years_of_experience": 12.6,
        "total_roles": 6,
        "current_role_months": 39,
        "production_ml_months": 54,
        "domains": ["fintech", "NLP"],
        "response_rate": 0.62,
        "notice_period_days": 30,
        "open_to_work": False,
        "disqualifier_flag": False,
        "disqualifier_areas": [],
        "honeypot_flag": False,
    }
    facts_2 = {
        "top_skills": ["LangChain", "OpenAI API"],
        "top_skill_months": [8, 6],
        "match_count": 4,
        "total_count": 10,
        "years_of_experience": 3.0,
        "total_roles": 2,
        "current_role_months": 14,
        "response_rate": 0.45,
        "notice_period_days": 60,
        "open_to_work": True,
        "disqualifier_flag": True,
        "disqualifier_areas": ["consultancy-only career", "no pre-LLM ML history"],
        "honeypot_flag": False,
    }
    facts_3 = {
        "honeypot_flag": True,
        "disqualifier_flag": False,
    }

    jd = None  # JDProfile unused at this stage

    print("=" * 70)
    print("REASONING GENERATOR -- COMPACT FORMAT SMOKE TEST")
    print("=" * 70)

    r1 = generate_reasoning(mock_breakdown_1, jd, rank=1,  candidate_facts=facts_1)
    r2 = generate_reasoning(mock_breakdown_2, jd, rank=45, candidate_facts=facts_2)
    r3 = generate_reasoning(mock_breakdown_3, jd, rank=95, candidate_facts=facts_3)

    print(f"\nRank 1  : {r1}")
    print(f"\nRank 45 : {r2}")
    print(f"\nRank 95 : {r3}")

    # ------------------------------------------------------------------
    # Real candidate from fixture (TEST_001)
    # ------------------------------------------------------------------
    if real_candidate is not None:
        mock_real = ScoreBreakdown(
            candidate_id=real_candidate["candidate_id"],
            skill_match=0.83,
            career_match=0.72,
            behavioral_score=0.78,
            availability_stability=0.70,
            seniority_and_shipping=0.80,
            disqualifier_penalty=0.0,
            availability_multiplier=1.05,
            base_score=0.810,
            final_score=0.851,
        )
        r_real = generate_reasoning(mock_real, jd, rank=3, candidate_facts=real_candidate)
        print(f"\nReal    : {r_real}")
        print(f"  (from fixture: {real_candidate['candidate_id']})")

    print()

    # ------------------------------------------------------------------
    # 4th call: verify JD-aware path with a mock JDProfile
    # ------------------------------------------------------------------
    class MockJD:
        must_have_skills = ["MLOps", "PyTorch", "Kubernetes", "RAG", "FAISS"]

    r_jd = generate_reasoning(
        mock_breakdown_1, MockJD(), rank=1, candidate_facts=facts_1
    )
    print(f"\nRank 1 (with JD): {r_jd}")
    print("  (should contain JD match/gap segment between opening and career)")
