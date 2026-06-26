"""
Stage 2 — Structured career-signal extraction.

Owner: Member A

Extracts per-candidate boolean/numeric signals from career_history, skills,
and education.  All keyword lists and thresholds are imported from config.py
so the team can tune during integration without touching this code.

Note: These are raw lexical signals only.  Semantic confirmation against the
JD happens downstream in src/ranking/fusion.py (Member C).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

import config
from src.features.schema import CandidateRecord, CareerEntry


# ---------------------------------------------------------------------------
# Pre-compiled regex patterns (built once at import time, not per-row)
# ---------------------------------------------------------------------------

def _build_keyword_pattern(keywords: list[str]) -> re.Pattern:
    """Build a case-insensitive regex that matches any keyword as a word."""
    # Escape special regex chars, join with | for alternation
    escaped = [re.escape(kw) for kw in keywords]
    # Use word boundary where possible, but allow partial for multi-word terms
    pattern = "|".join(escaped)
    return re.compile(pattern, re.IGNORECASE)


_PRODUCTION_RE = _build_keyword_pattern(config.PRODUCTION_KEYWORDS)
_RESEARCH_RE = _build_keyword_pattern(config.RESEARCH_KEYWORDS)
_SENIORITY_RE = _build_keyword_pattern(config.SENIORITY_TITLE_KEYWORDS)
_MANAGEMENT_RE = _build_keyword_pattern(config.MANAGEMENT_TITLE_KEYWORDS)
_HANDS_ON_RE = _build_keyword_pattern(config.HANDS_ON_ENGINEERING_KEYWORDS)
_LLM_GENAI_RE = _build_keyword_pattern(config.LLM_GENAI_SKILL_KEYWORDS)
_DEEP_ML_RE = _build_keyword_pattern(config.DEEP_ML_SKILL_KEYWORDS)
_CV_SPEECH_ROBOTICS_RE = _build_keyword_pattern(config.CV_SPEECH_ROBOTICS_KEYWORDS)
_NLP_IR_LLM_RE = _build_keyword_pattern(config.NLP_IR_LLM_KEYWORDS)


# ---------------------------------------------------------------------------
# Employer type classification
# ---------------------------------------------------------------------------

@lru_cache(maxsize=2048)
def classify_industry(industry: str) -> str:
    """
    Classify an industry string into {product, services, research, other}.

    Uses the config-driven INDUSTRY_TO_EMPLOYER_TYPE map (first substring
    match wins).  Results are cached since many candidates share the same
    industry strings.
    """
    if not industry:
        return config.EMPLOYER_TYPE_DEFAULT

    industry_lower = industry.lower().strip()

    for pattern, employer_type in config.INDUSTRY_TO_EMPLOYER_TYPE.items():
        if pattern in industry_lower:
            return employer_type

    return config.EMPLOYER_TYPE_DEFAULT


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CareerSignals:
    """
    Container for all extracted career signals for one candidate.

    Every field is a float in [0, 1] or a boolean flag.  Downstream stages
    (fusion.py, behavioral.py) consume these directly.
    """
    candidate_id: str = ""

    # Employer type for current role
    employer_type_current: str = "other"

    # Per-career-entry employer types (for consulting_only computation)
    employer_types: list[str] = field(default_factory=list)

    # Flags and scores
    consulting_only_flag: bool = False
    production_evidence_score: float = 0.0      # 0–1
    title_chaser_score: float = 0.0             # 0–1
    research_only_flag: bool = False
    recent_llm_only_flag: bool = False
    cv_speech_without_nlp_flag: bool = False
    stale_coder_flag: bool = False

    # Raw derived fields
    total_career_months: int = 0
    seniority_score: float = 0.0                # 0–1
    shipping_score: float = 0.0                 # alias for production_evidence_score


# ---------------------------------------------------------------------------
# Signal extractors
# ---------------------------------------------------------------------------

def _compute_employer_types(candidate: CandidateRecord) -> tuple[str, list[str]]:
    """Classify employer type for current industry and all career entries."""
    current_type = classify_industry(candidate.current_industry)

    entry_types: list[str] = []
    for entry in candidate.career_history:
        entry_types.append(classify_industry(entry.industry))

    return current_type, entry_types


def _compute_consulting_only(
    employer_types: list[str],
    years_of_experience: float,
) -> bool:
    """
    True if every career entry is "services", with zero "product" entries,
    AND total experience >= 3 years.  Junior candidates with one services
    internship aren't flagged the same way.
    """
    if not employer_types:
        return False
    if years_of_experience < 3.0:
        return False

    has_product = any(t == "product" for t in employer_types)
    all_services = all(t == "services" for t in employer_types)

    return all_services and not has_product


def _compute_production_evidence(career_history: list[CareerEntry]) -> float:
    """
    Scan career_history descriptions for production/deployment keywords.
    Weight more recent roles higher (current role gets full weight, older
    roles decay).  Returns a score in [0, 1].
    """
    if not career_history:
        return 0.0

    # Sort by start_date descending (most recent first)
    sorted_entries = sorted(
        career_history,
        key=lambda e: e.start_date or "",
        reverse=True,
    )

    total_weight = 0.0
    weighted_hits = 0.0

    for i, entry in enumerate(sorted_entries):
        if entry.is_current:
            weight = config.PRODUCTION_RECENCY_WEIGHT_CURRENT
        else:
            weight = max(
                0.1,
                config.PRODUCTION_RECENCY_WEIGHT_CURRENT
                - i * config.PRODUCTION_RECENCY_DECAY_PER_POSITION,
            )

        total_weight += weight

        desc = entry.description or ""
        if desc and _PRODUCTION_RE.search(desc):
            # Count distinct keyword matches for richer signal
            matches = set(_PRODUCTION_RE.findall(desc.lower()))
            # More distinct matches → stronger signal (cap at 1.0 per entry)
            match_density = min(1.0, len(matches) / 5.0)
            weighted_hits += weight * match_density

    if total_weight == 0.0:
        return 0.0

    return min(1.0, weighted_hits / total_weight)


def _compute_title_chaser(career_history: list[CareerEntry]) -> float:
    """
    Detect rapid title escalation combined with short tenures.

    Returns a score in [0, 1]:
      - 0.0 = no title-chasing signal
      - 1.0 = strong title-chasing pattern
    """
    if len(career_history) < config.TITLE_CHASER_MIN_EMPLOYERS:
        return 0.0

    # Compute unique employers
    unique_employers = set()
    tenures: list[int] = []
    seniority_levels: list[int] = []

    for entry in career_history:
        company_key = (entry.company or "").strip().lower()
        if company_key:
            unique_employers.add(company_key)
        tenures.append(entry.duration_months or 0)

        # Compute seniority level from title keywords
        title_lower = (entry.title or "").lower()
        level = 0
        for j, kw in enumerate(config.SENIORITY_TITLE_KEYWORDS):
            if kw in title_lower:
                # Higher index = lower seniority in our list ordering
                # (chief > director > head > lead > senior > ...)
                level = max(level, len(config.SENIORITY_TITLE_KEYWORDS) - j)
        seniority_levels.append(level)

    if len(unique_employers) < config.TITLE_CHASER_MIN_EMPLOYERS:
        return 0.0

    # Average tenure
    avg_tenure = sum(tenures) / len(tenures) if tenures else 0
    if avg_tenure >= config.TITLE_CHASER_MAX_AVG_TENURE_MONTHS:
        return 0.0

    # Check for monotonically increasing seniority
    if len(seniority_levels) < 2:
        return 0.0

    increases = 0
    for k in range(1, len(seniority_levels)):
        if seniority_levels[k] > seniority_levels[k - 1]:
            increases += 1

    if increases == 0:
        return 0.0

    # Score: combine short tenure ratio with seniority escalation rate
    tenure_signal = max(
        0.0,
        1.0 - avg_tenure / config.TITLE_CHASER_MAX_AVG_TENURE_MONTHS,
    )
    escalation_signal = increases / (len(seniority_levels) - 1)

    return min(1.0, tenure_signal * 0.5 + escalation_signal * 0.5)


def _compute_research_only(
    candidate: CandidateRecord,
    production_score: float,
    employer_types: list[str],
) -> bool:
    """
    True if descriptions/titles are dense in research terms and have zero
    production evidence AND zero "product" employer entries.
    """
    if production_score > 0.0:
        return False

    if any(t == "product" for t in employer_types):
        return False

    # Check if career descriptions and titles are research-heavy
    research_hits = 0
    total_entries = max(1, len(candidate.career_history))

    for entry in candidate.career_history:
        text = f"{entry.title} {entry.description}".lower()
        if _RESEARCH_RE.search(text):
            research_hits += 1

    # Also check summary/headline
    profile_text = f"{candidate.headline} {candidate.summary}".lower()
    if _RESEARCH_RE.search(profile_text):
        research_hits += 1
        total_entries += 1

    return research_hits / total_entries >= 0.5


def _compute_recent_llm_only(candidate: CandidateRecord) -> bool:
    """
    True if the only skills matching LLM/GenAI keywords have
    duration_months < 12, AND there is no other ML/DS skill with
    duration_months >= 24.

    Catches "wrapped GPT-4 in a chatbot for 6 months" candidates,
    not genuinely deep ML people who are also new to LLMs.
    """
    has_llm_skill = False
    has_seasoned_llm = False
    has_deep_ml = False

    for skill in candidate.skills:
        name_lower = skill.name.lower()

        is_llm = bool(_LLM_GENAI_RE.search(name_lower))
        is_deep_ml = bool(_DEEP_ML_RE.search(name_lower))

        if is_llm:
            has_llm_skill = True
            if skill.duration_months >= config.RECENT_LLM_ONLY_MAX_DURATION_MONTHS:
                has_seasoned_llm = True

        if is_deep_ml and skill.duration_months >= config.DEEP_ML_MIN_DURATION_MONTHS:
            has_deep_ml = True

    # Flag only if they have LLM skills, all are recent, and no deep ML background
    return has_llm_skill and not has_seasoned_llm and not has_deep_ml


def _compute_cv_speech_without_nlp(candidate: CandidateRecord) -> bool:
    """
    True if skills are dominated by CV/Speech/Robotics keywords with zero
    NLP/IR/LLM/Search keywords anywhere in skills or descriptions.
    """
    has_cv_speech = False
    has_nlp = False

    # Check skills
    for skill in candidate.skills:
        name_lower = skill.name.lower()
        if _CV_SPEECH_ROBOTICS_RE.search(name_lower):
            has_cv_speech = True
        if _NLP_IR_LLM_RE.search(name_lower):
            has_nlp = True

    if has_nlp:
        return False
    if not has_cv_speech:
        return False

    # Also check descriptions and summary for NLP keywords
    all_text_parts = [candidate.headline, candidate.summary]
    for entry in candidate.career_history:
        all_text_parts.append(entry.description)
        all_text_parts.append(entry.title)

    combined_text = " ".join(all_text_parts).lower()
    if _NLP_IR_LLM_RE.search(combined_text):
        return False

    return True


def _compute_stale_coder(candidate: CandidateRecord) -> bool:
    """
    True if current_title matches a management keyword, the current role's
    description has no hands-on engineering keywords, AND
    years_of_experience >= STALE_CODER_MIN_EXPERIENCE_YEARS.
    """
    if candidate.years_of_experience < config.STALE_CODER_MIN_EXPERIENCE_YEARS:
        return False

    title_lower = candidate.current_title.lower()
    if not _MANAGEMENT_RE.search(title_lower):
        return False

    # Find current role description
    current_desc = ""
    for entry in candidate.career_history:
        if entry.is_current:
            current_desc = entry.description or ""
            break

    if not current_desc:
        # No current role description → can't confirm stale, be conservative
        return False

    return not bool(_HANDS_ON_RE.search(current_desc.lower()))


def _compute_seniority_score(candidate: CandidateRecord) -> float:
    """
    Compute a seniority score in [0, 1] based on current title and
    years of experience.
    """
    score = 0.0

    title_lower = candidate.current_title.lower()
    # Check against seniority keywords, higher match = higher score
    for i, kw in enumerate(config.SENIORITY_TITLE_KEYWORDS):
        if kw in title_lower:
            # Keywords are ordered by seniority (chief first, senior last)
            score = max(
                score,
                (len(config.SENIORITY_TITLE_KEYWORDS) - i)
                / len(config.SENIORITY_TITLE_KEYWORDS),
            )

    # Blend with experience-based seniority (cap at 20 years)
    exp_score = min(1.0, candidate.years_of_experience / 20.0)
    return min(1.0, score * 0.6 + exp_score * 0.4)


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_career_signals(candidate: CandidateRecord) -> CareerSignals:
    """
    Analyse a single candidate's career_history, skills, and education to
    produce structured signals.

    Parameters
    ----------
    candidate : CandidateRecord
        Typed candidate record (output of ``parse_candidate``).

    Returns
    -------
    CareerSignals
        Extracted signal container.
    """
    # Employer types
    current_type, entry_types = _compute_employer_types(candidate)

    # Total career months
    total_months = sum(e.duration_months for e in candidate.career_history)

    # Production evidence
    production_score = _compute_production_evidence(candidate.career_history)

    # Title chaser
    title_chaser = _compute_title_chaser(candidate.career_history)

    # Research only
    research_only = _compute_research_only(candidate, production_score, entry_types)

    # Recent LLM only
    recent_llm_only = _compute_recent_llm_only(candidate)

    # CV/Speech without NLP
    cv_speech_no_nlp = _compute_cv_speech_without_nlp(candidate)

    # Stale coder
    stale_coder = _compute_stale_coder(candidate)

    # Seniority
    seniority = _compute_seniority_score(candidate)

    # Consulting only
    consulting_only = _compute_consulting_only(
        entry_types, candidate.years_of_experience
    )

    return CareerSignals(
        candidate_id=candidate.candidate_id,
        employer_type_current=current_type,
        employer_types=entry_types,
        consulting_only_flag=consulting_only,
        production_evidence_score=production_score,
        title_chaser_score=title_chaser,
        research_only_flag=research_only,
        recent_llm_only_flag=recent_llm_only,
        cv_speech_without_nlp_flag=cv_speech_no_nlp,
        stale_coder_flag=stale_coder,
        total_career_months=total_months,
        seniority_score=seniority,
        shipping_score=production_score,  # alias
    )
