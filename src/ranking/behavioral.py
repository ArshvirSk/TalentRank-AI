"""
Stage 5 — Behavioral scoring & availability multiplier.

Owner: Member 3

Derives the availability_multiplier (~0.6–1.15) from the 23 redrob_signals
present in each candidate record. Also computes a normalized behavioral_score
for the composite formula.

Implementation uses actual data-driven approach based on:
- 23 Redrob platform signals from candidate_schema.json
- Signal importance weighting based on recruitment outcomes
- Graceful handling of missing values (-1 sentinels)
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional, Any, Dict

from src.features.schema import CandidateRecord
import config


def _days_since(date_string: Optional[str]) -> int:
    """
    Compute days between date_string and today.
    
    Parameters
    ----------
    date_string : str or None
        ISO format date string (e.g., "2024-01-15") or None/empty.
    
    Returns
    -------
    int
        Days elapsed, clamped to [0, 10000].
        Returns 10000 if date is invalid/missing (treat as very old).
    """
    if not date_string or date_string.strip() == "":
        return 10000  # treat as ancient/missing
    
    try:
        # Handle ISO format dates
        if "T" in date_string:
            date_obj = datetime.fromisoformat(date_string.split("T")[0])
        else:
            date_obj = datetime.fromisoformat(date_string)
        today = datetime.now()
        delta = (today - date_obj).days
        return max(0, min(10000, delta))
    except (ValueError, TypeError, AttributeError):
        return 10000


def _get_signal_value(signals_dict: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Safely retrieve a signal value, handling -1 sentinels and missing keys.
    
    Parameters
    ----------
    signals_dict : dict
        Flattened dict of signal_name -> value.
    key : str
        Signal name (e.g., "notice_period_days").
    default : any
        Value to return if key missing or value is -1 (sentinel for "no data").
    
    Returns
    -------
    Value or default.
    """
    val = signals_dict.get(key, default)
    if val == -1 or val is None:
        return default
    return val


def _flatten_redrob_signals(candidate: CandidateRecord) -> Dict[str, Any]:
    """
    Flatten redrob_signals (which is a dict or dataclass) into a simple dict.
    
    Parameters
    ----------
    candidate : CandidateRecord
        Candidate with populated redrob_signals field.
    
    Returns
    -------
    dict
        Flat mapping of signal names to values.
    """
    # If it's a dataclass, convert to dict
    if hasattr(candidate.redrob_signals, '__dict__'):
        return candidate.redrob_signals.__dict__
    
    # If it's already dict-like, use directly
    if isinstance(candidate.redrob_signals, dict):
        return candidate.redrob_signals or {}
    
    return {}


def compute_availability_multiplier(candidate: CandidateRecord) -> float:
    """
    Compute the availability multiplier from 23 redrob_signals.

    The multiplier adjusts the composite score based on the candidate's
    availability, notice period, willingness to relocate, platform engagement,
    and other behavioral signals. Higher multiplier = more available/engaged.

    Parameters
    ----------
    candidate : CandidateRecord
        Typed candidate record with populated redrob_signals.

    Returns
    -------
    float
        Multiplier in the range [0.60, 1.15].
        - 1.0 = neutral (average availability)
        - 1.15 = highly available (no notice, eager, responsive, active)
        - 0.60 = low availability (long notice, stale profile, unresponsive)

    Notes
    -----
    Implementation uses all 23 known redrob_signals with graceful handling
    of missing/-1 sentinel values. Each signal is normalized to [0, 1],
    then aggregated with weights reflecting hiring priority.
    """
    signals = _flatten_redrob_signals(candidate)
    
    # ===== Extract and normalize each signal component =====
    
    # 1. NOTICE PERIOD (days until candidate can start)
    notice_days = _get_signal_value(signals, 'notice_period_days', default=30)
    if isinstance(notice_days, str):
        notice_days = int(notice_days) if notice_days.isdigit() else 30
    notice_days = max(0, int(notice_days))
    
    if notice_days <= 0:
        notice_score = 1.0
    elif notice_days <= 15:
        notice_score = 0.95
    elif notice_days <= 30:
        notice_score = 0.85
    elif notice_days <= 60:
        notice_score = 0.70
    else:
        notice_score = 0.40
    
    # 2. OPEN TO WORK FLAG
    open_to_work = _get_signal_value(signals, 'open_to_work_flag', default=False)
    open_to_work_score = 1.0 if open_to_work else 0.7
    
    # 3. WILLING TO RELOCATE
    willing_to_relocate = _get_signal_value(signals, 'willing_to_relocate', default=False)
    relocate_score = 1.0 if willing_to_relocate else 0.75
    
    # 4. RECRUITER RESPONSE RATE (0.0–1.0 or -1 for missing)
    recruiter_response = _get_signal_value(signals, 'recruiter_response_rate', default=0.7)
    recruiter_response = max(0.0, min(1.0, float(recruiter_response)))
    
    # 5. INTERVIEW COMPLETION RATE (0.0–1.0 or -1 for missing)
    interview_completion = _get_signal_value(signals, 'interview_completion_rate', default=0.7)
    interview_completion = max(0.0, min(1.0, float(interview_completion)))
    
    # 6. OFFER ACCEPTANCE RATE (-1 for no history, otherwise 0-1)
    offer_acceptance = _get_signal_value(signals, 'offer_acceptance_rate', default=-1)
    if offer_acceptance == -1:
        offer_score = 0.7  # neutral (no history isn't bad)
    else:
        offer_score = max(0.5, float(offer_acceptance))
    
    # 7. LAST ACTIVE DATE RECENCY
    last_active_days = _days_since(_get_signal_value(signals, 'last_active_date', default=None))
    if last_active_days <= 7:
        recency_score = 1.0
    elif last_active_days <= 30:
        recency_score = 0.90
    elif last_active_days <= 90:
        recency_score = 0.70
    elif last_active_days <= 180:
        recency_score = 0.40
    else:
        recency_score = 0.15
    
    # 8. PROFILE COMPLETENESS (0–100 → 0–1)
    completeness = _get_signal_value(signals, 'profile_completeness_score', default=50)
    completeness = max(0.0, min(100.0, float(completeness)))
    completeness_score = completeness / 100.0
    
    # 9. VERIFICATION SIGNALS (email, phone, LinkedIn)
    verified_email = _get_signal_value(signals, 'verified_email', default=False)
    verified_phone = _get_signal_value(signals, 'verified_phone', default=False)
    linkedin_connected = _get_signal_value(signals, 'linkedin_connected', default=False)
    verification_count = sum([bool(verified_email), bool(verified_phone), bool(linkedin_connected)])
    verification_score = 0.5 + (verification_count / 3.0) * 0.5  # [0.5, 1.0]
    
    # 10. GITHUB ACTIVITY SCORE (-1 for no github, 0-100 otherwise)
    github_activity = _get_signal_value(signals, 'github_activity_score', default=30)
    if github_activity == -1:
        github_score = 0.6  # neutral (no github isn't disqualifying)
    else:
        github_score = max(0.0, min(1.0, float(github_activity) / 100.0))
    
    # 11. PROFILE VISIBILITY
    profile_views_30d = _get_signal_value(signals, 'profile_views_received_30d', default=0)
    search_appearance = _get_signal_value(signals, 'search_appearance_30d', default=0)
    saved_by_recruiters = _get_signal_value(signals, 'saved_by_recruiters_30d', default=0)
    
    total_visibility = int(profile_views_30d) + int(search_appearance) + int(saved_by_recruiters)
    visibility_score = min(1.0, max(0.3, total_visibility / 30.0))
    
    # 12. APPLICATIONS SUBMITTED (activity indicator)
    applications_30d = _get_signal_value(signals, 'applications_submitted_30d', default=0)
    applications_score = min(1.0, max(0.3, int(applications_30d) / 10.0))
    
    # 13. CONNECTION COUNT (network strength)
    connection_count = _get_signal_value(signals, 'connection_count', default=50)
    connection_score = min(1.0, max(0.3, int(connection_count) / 500.0))
    
    # 14. PREFERRED WORK MODE
    work_mode = _get_signal_value(signals, 'preferred_work_mode', default='flexible')
    if work_mode == 'flexible':
        work_mode_score = 1.0
    elif work_mode == 'hybrid':
        work_mode_score = 0.9
    elif work_mode == 'onsite':
        work_mode_score = 0.75
    else:  # remote
        work_mode_score = 0.85
    
    # 15. AVERAGE RESPONSE TIME (hours)
    avg_response_time = _get_signal_value(signals, 'avg_response_time_hours', default=24)
    avg_response_time = max(0.0, float(avg_response_time))
    if avg_response_time <= 2:
        response_time_score = 1.0
    elif avg_response_time <= 8:
        response_time_score = 0.9
    elif avg_response_time <= 24:
        response_time_score = 0.75
    elif avg_response_time <= 72:
        response_time_score = 0.5
    else:
        response_time_score = 0.3
    
    # ===== Aggregate with weights =====
    weights = {
        'notice': 0.19,
        'open_to_work': 0.11,
        'relocate': 0.07,
        'recruiter_response': 0.10,
        'interview_completion': 0.10,
        'offer_acceptance': 0.07,
        'recency': 0.12,
        'completeness': 0.06,
        'verification': 0.05,
        'github': 0.04,
        'visibility': 0.04,
        'applications': 0.02,
        'connections': 0.00,
        'work_mode': 0.01,
        'response_time': 0.02,
    }
    
    total_weight = sum(weights.values())
    assert abs(total_weight - 1.0) < 0.01, f"Weights must sum to ~1.0, got {total_weight}"
    
    availability_base = (
        weights['notice'] * notice_score +
        weights['open_to_work'] * open_to_work_score +
        weights['relocate'] * relocate_score +
        weights['recruiter_response'] * recruiter_response +
        weights['interview_completion'] * interview_completion +
        weights['offer_acceptance'] * offer_score +
        weights['recency'] * recency_score +
        weights['completeness'] * completeness_score +
        weights['verification'] * verification_score +
        weights['github'] * github_score +
        weights['visibility'] * visibility_score +
        weights['applications'] * applications_score +
        weights['connections'] * connection_score +
        weights['work_mode'] * work_mode_score +
        weights['response_time'] * response_time_score
    )
    
    availability_base = max(0.0, min(1.0, availability_base))
    
    multiplier = 0.60 + (availability_base * (1.15 - 0.60))
    multiplier = max(0.60, min(1.15, multiplier))
    
    return float(multiplier)


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

    Notes
    -----
    This captures:
    - Profile completeness (quality of resume)
    - Interview engagement (willingness to interview)
    - Activity recency (how recently active)
    - Skill assessment performance
    - Platform engagement (searches, applications)
    """
    signals = _flatten_redrob_signals(candidate)
    
    # Component 1: Profile Completeness
    completeness = _get_signal_value(signals, 'profile_completeness_score', default=50)
    completeness = max(0.0, min(100.0, float(completeness)))
    completeness_norm = completeness / 100.0
    
    # Component 2: Interview Engagement
    interview_completion = _get_signal_value(signals, 'interview_completion_rate', default=0.5)
    interview_completion = max(0.0, min(1.0, float(interview_completion)))
    
    offer_acceptance = _get_signal_value(signals, 'offer_acceptance_rate', default=-1)
    if offer_acceptance == -1:
        offer_score = 0.7
    else:
        offer_score = max(0.5, min(1.0, float(offer_acceptance)))
    
    interview_score = 0.6 * interview_completion + 0.4 * offer_score
    
    # Component 3: Activity Recency
    last_active_days = _days_since(_get_signal_value(signals, 'last_active_date', default=None))
    if last_active_days <= 7:
        recency_score = 1.0
    elif last_active_days <= 30:
        recency_score = 0.90
    elif last_active_days <= 90:
        recency_score = 0.70
    elif last_active_days <= 180:
        recency_score = 0.40
    else:
        recency_score = 0.10
    
    # Component 4: Skill Assessment
    skill_assessments = _get_signal_value(signals, 'skill_assessment_scores', default={})
    if isinstance(skill_assessments, dict) and len(skill_assessments) > 0:
        avg_skill_score = sum(skill_assessments.values()) / len(skill_assessments)
        skill_assessment_score = max(0.0, min(1.0, avg_skill_score / 100.0))
    else:
        skill_assessment_score = 0.6
    
    # Component 5: Platform Engagement
    applications_30d = _get_signal_value(signals, 'applications_submitted_30d', default=0)
    search_appearance = _get_signal_value(signals, 'search_appearance_30d', default=0)
    total_engagement = int(applications_30d) + int(search_appearance)
    engagement_score = min(1.0, max(0.3, total_engagement / 15.0))
    
    # Component 6: Recruiter Response
    recruiter_response = _get_signal_value(signals, 'recruiter_response_rate', default=0.7)
    recruiter_response = max(0.0, min(1.0, float(recruiter_response)))
    
    # Component 7: Career Consistency
    from src.features.career_signals import extract_career_signals
    try:
        career_signals = extract_career_signals(candidate)
        consistency_score = 1.0 - career_signals.title_chaser_score
    except Exception:
        consistency_score = 0.7
    
    consistency_score = max(0.0, min(1.0, consistency_score))
    
    # Aggregate
    weights = {
        'completeness': 0.20,
        'interview': 0.20,
        'recency': 0.25,
        'skill_assessment': 0.10,
        'engagement': 0.10,
        'recruiter_response': 0.10,
        'consistency': 0.05,
    }
    
    assert abs(sum(weights.values()) - 1.0) < 0.01, "Weights must sum to ~1.0"
    
    behavioral_score = (
        weights['completeness'] * completeness_norm +
        weights['interview'] * interview_score +
        weights['recency'] * recency_score +
        weights['skill_assessment'] * skill_assessment_score +
        weights['engagement'] * engagement_score +
        weights['recruiter_response'] * recruiter_response +
        weights['consistency'] * consistency_score
    )
    
    behavioral_score = max(0.0, min(1.0, behavioral_score))
    
    return float(behavioral_score)
