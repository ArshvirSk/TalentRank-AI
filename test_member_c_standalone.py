#!/usr/bin/env python
"""
Standalone test script for Member C implementations.
Tests behavioral.py, fusion.py functions without numpy/pytest overhead.
"""

import sys
sys.path.insert(0, '.')

from datetime import datetime, timedelta
from src.features.schema import CandidateRecord, CareerEntry, SkillEntry, RedrobSignals
from src.features.career_signals import CareerSignals
from src.ranking.behavioral import (
    compute_availability_multiplier,
    compute_behavioral_score,
    _days_since,
)
from src.ranking.fusion import (
    compute_composite_score,
    ScoreBreakdown,
)


def create_perfect_candidate():
    """Create a candidate with all perfect signals."""
    signals = RedrobSignals(
        profile_completeness_score=100,
        verified_email=True,
        verified_phone=True,
        linkedin_connected=True,
        signup_date="2018-01-01",
        last_active_date=(datetime.now() - timedelta(days=1)).isoformat(),
    )
    
    # Add extra signals as dict
    signals_dict = {
        "profile_completeness_score": 100,
        "signup_date": "2018-01-01",
        "last_active_date": (datetime.now() - timedelta(days=1)).isoformat(),
        "open_to_work_flag": True,
        "profile_views_received_30d": 50,
        "applications_submitted_30d": 10,
        "recruiter_response_rate": 0.95,
        "avg_response_time_hours": 1,
        "skill_assessment_scores": {"Python": 95},
        "connection_count": 1000,
        "endorsements_received": 100,
        "notice_period_days": 0,
        "expected_salary_range_inr_lpa": {"min": 30, "max": 50},
        "preferred_work_mode": "flexible",
        "willing_to_relocate": True,
        "github_activity_score": 90,
        "search_appearance_30d": 30,
        "saved_by_recruiters_30d": 20,
        "interview_completion_rate": 1.0,
        "offer_acceptance_rate": 1.0,
        "verified_email": True,
        "verified_phone": True,
        "linkedin_connected": True,
    }
    
    candidate = CandidateRecord(
        candidate_id="CAND_0000001",
        years_of_experience=10.0,
        current_title="Senior Engineer",
        current_company="FAANG",
        current_company_size="10001+",
        current_industry="Tech",
        headline="Senior Engineer",
        summary="10+ years experience",
        location="Mumbai",
        career_history=[
            CareerEntry(
                company="FAANG",
                title="Senior",
                start_date="2022-01-01",
                end_date="",
                duration_months=24,
                is_current=True,
                industry="Tech",
                company_size="10001+",
                description="Leadership",
            ),
        ],
        education=[],
        skills=[SkillEntry(name="Python", proficiency="expert", endorsements=50, duration_months=120)],
        redrob_signals=signals_dict,
    )
    
    return candidate


def create_poor_candidate():
    """Create a candidate with all poor signals."""
    signals_dict = {
        "profile_completeness_score": 30,
        "signup_date": "2023-06-01",
        "last_active_date": (datetime.now() - timedelta(days=200)).isoformat(),
        "open_to_work_flag": False,
        "profile_views_received_30d": 0,
        "applications_submitted_30d": 0,
        "recruiter_response_rate": 0.1,
        "avg_response_time_hours": 120,
        "skill_assessment_scores": {},
        "connection_count": 10,
        "endorsements_received": 0,
        "notice_period_days": 90,
        "expected_salary_range_inr_lpa": {"min": 5, "max": 8},
        "preferred_work_mode": "remote",
        "willing_to_relocate": False,
        "github_activity_score": -1,
        "search_appearance_30d": 0,
        "saved_by_recruiters_30d": 0,
        "interview_completion_rate": 0.2,
        "offer_acceptance_rate": -1,
        "verified_email": False,
        "verified_phone": False,
        "linkedin_connected": False,
    }
    
    candidate = CandidateRecord(
        candidate_id="CAND_0000002",
        years_of_experience=0.5,
        current_title="Junior",
        current_company="Startup",
        current_company_size="1-10",
        current_industry="Tech",
        headline="Junior Developer",
        summary="Beginner",
        location="Remote",
        career_history=[],
        education=[],
        skills=[],
        redrob_signals=signals_dict,
    )
    
    return candidate


def test_days_since():
    """Test _days_since utility."""
    print("\n=== Testing _days_since ===")
    
    # Test None
    result = _days_since(None)
    assert result == 10000, f"Expected 10000 for None, got {result}"
    print("[PASS] _days_since(None) = 10000")
    
    # Test empty string
    result = _days_since("")
    assert result == 10000, f"Expected 10000 for empty string, got {result}"
    print("[PASS] _days_since('') = 10000")
    
    # Test recent date
    yesterday = (datetime.now() - timedelta(days=1)).isoformat()
    result = _days_since(yesterday)
    assert 0 <= result <= 2, f"Expected [0,2] for yesterday, got {result}"
    print(f"[PASS] _days_since(yesterday) = {result}")


def test_availability_multiplier():
    """Test compute_availability_multiplier."""
    print("\n=== Testing compute_availability_multiplier ===")
    
    perfect = create_perfect_candidate()
    multiplier_perfect = compute_availability_multiplier(perfect)
    assert 1.0 <= multiplier_perfect <= 1.15, f"Perfect: expected [1.0,1.15], got {multiplier_perfect}"
    print(f"[PASS] Perfect candidate multiplier: {multiplier_perfect:.3f} in [1.0, 1.15]")
    
    poor = create_poor_candidate()
    multiplier_poor = compute_availability_multiplier(poor)
    assert 0.60 <= multiplier_poor <= 0.85, f"Poor: expected [0.60,0.85], got {multiplier_poor}"
    print(f"[PASS] Poor candidate multiplier: {multiplier_poor:.3f} in [0.60, 0.85]")
    
    assert multiplier_perfect > multiplier_poor, "Perfect should have higher multiplier than poor"
    print(f"[PASS] Perfect ({multiplier_perfect:.3f}) > Poor ({multiplier_poor:.3f})")


def test_behavioral_score():
    """Test compute_behavioral_score."""
    print("\n=== Testing compute_behavioral_score ===")
    
    perfect = create_perfect_candidate()
    score_perfect = compute_behavioral_score(perfect)
    assert 0.0 <= score_perfect <= 1.0, f"Expected [0,1], got {score_perfect}"
    assert score_perfect >= 0.7, f"Perfect should be high (>=0.7), got {score_perfect}"
    print(f"✓ Perfect candidate score: {score_perfect:.3f} ∈ [0.7, 1.0]")
    
    poor = create_poor_candidate()
    score_poor = compute_behavioral_score(poor)
    assert 0.0 <= score_poor <= 1.0, f"Expected [0,1], got {score_poor}"
    assert score_poor <= 0.4, f"Poor should be low (<=0.4), got {score_poor}"
    print(f"✓ Poor candidate score: {score_poor:.3f} ∈ [0.0, 0.4]")
    
    assert score_perfect > score_poor, "Perfect should have higher behavioral score"
    print(f"✓ Perfect ({score_perfect:.3f}) > Poor ({score_poor:.3f})")


def test_composite_score():
    """Test compute_composite_score."""
    print("\n=== Testing compute_composite_score ===")
    
    career_signals = CareerSignals(
        candidate_id="CAND_TEST",
        employer_type_current="tech",
        consulting_only_flag=False,
        production_evidence_score=0.9,
        title_chaser_score=0.1,
        research_only_flag=False,
        recent_llm_only_flag=False,
        cv_speech_without_nlp_flag=False,
        stale_coder_flag=False,
        total_career_months=120,
        seniority_score=0.95,
        shipping_score=0.9,
    )
    
    breakdown = compute_composite_score(
        candidate_id="CAND_TEST",
        skill_match_sim=0.9,
        career_match_sim=0.85,
        disqualifier_sim=0.1,
        career_signals=career_signals,
        availability_multiplier=1.1,
        behavioral_score=0.8,
    )
    
    assert isinstance(breakdown, ScoreBreakdown), "Should return ScoreBreakdown"
    assert 0.0 <= breakdown.final_score <= 1.0, f"Final score out of range: {breakdown.final_score}"
    assert breakdown.final_score > 0.5, f"High-quality candidate should score >0.5, got {breakdown.final_score}"
    print(f"✓ Composite score: {breakdown.final_score:.3f} ∈ [0.0, 1.0]")
    print(f"  - Skill match: {breakdown.skill_match:.3f}")
    print(f"  - Career match: {breakdown.career_match:.3f}")
    print(f"  - Behavioral: {breakdown.behavioral_score:.3f}")
    print(f"  - Base: {breakdown.base_score:.3f}")


def test_disqualifier_penalty():
    """Test that disqualifier penalty works."""
    print("\n=== Testing disqualifier penalty ===")
    
    career_signals = CareerSignals(
        candidate_id="CAND_1",
        employer_type_current="tech",
        consulting_only_flag=False,
        production_evidence_score=0.9,
        title_chaser_score=0.1,
        research_only_flag=False,
        recent_llm_only_flag=False,
        cv_speech_without_nlp_flag=False,
        stale_coder_flag=False,
        total_career_months=120,
        seniority_score=0.95,
        shipping_score=0.9,
    )
    
    # Without disqualifier
    breakdown_no_disq = compute_composite_score(
        candidate_id="CAND_1",
        skill_match_sim=0.8,
        career_match_sim=0.7,
        disqualifier_sim=0.1,
        career_signals=career_signals,
        availability_multiplier=1.0,
        behavioral_score=0.7,
    )
    
    # With high disqualifier
    breakdown_high_disq = compute_composite_score(
        candidate_id="CAND_2",
        skill_match_sim=0.8,
        career_match_sim=0.7,
        disqualifier_sim=0.8,
        career_signals=career_signals,
        availability_multiplier=1.0,
        behavioral_score=0.7,
    )
    
    assert breakdown_no_disq.final_score > breakdown_high_disq.final_score, \
        f"Low disq ({breakdown_no_disq.final_score:.3f}) should > high disq ({breakdown_high_disq.final_score:.3f})"
    print(f"✓ Low disqualifier: {breakdown_no_disq.final_score:.3f}")
    print(f"✓ High disqualifier: {breakdown_high_disq.final_score:.3f}")
    print(f"  (High disqualifier penalty: {breakdown_high_disq.disqualifier_penalty:.3f})")


def test_multiplier_impact():
    """Test that availability multiplier correctly scales score."""
    print("\n=== Testing availability multiplier impact ===")
    
    career_signals = CareerSignals(
        candidate_id="CAND_1",
        employer_type_current="tech",
        consulting_only_flag=False,
        production_evidence_score=0.6,
        title_chaser_score=0.4,
        research_only_flag=False,
        recent_llm_only_flag=False,
        cv_speech_without_nlp_flag=False,
        stale_coder_flag=False,
        total_career_months=60,
        seniority_score=0.6,
        shipping_score=0.6,
    )
    
    # Low multiplier
    breakdown_low = compute_composite_score(
        candidate_id="CAND_1",
        skill_match_sim=0.7,
        career_match_sim=0.6,
        disqualifier_sim=0.1,
        career_signals=career_signals,
        availability_multiplier=0.60,
        behavioral_score=0.7,
    )
    
    # High multiplier
    breakdown_high = compute_composite_score(
        candidate_id="CAND_2",
        skill_match_sim=0.7,
        career_match_sim=0.6,
        disqualifier_sim=0.1,
        career_signals=career_signals,
        availability_multiplier=1.15,
        behavioral_score=0.7,
    )
    
    ratio = breakdown_high.final_score / breakdown_low.final_score
    assert 1.5 <= ratio <= 2.0, f"Multiplier ratio should be ~1.91x, got {ratio:.2f}x"
    print(f"✓ Low multiplier (0.60) score: {breakdown_low.final_score:.3f}")
    print(f"✓ High multiplier (1.15) score: {breakdown_high.final_score:.3f}")
    print(f"✓ Ratio: {ratio:.2f}x (expected ~1.91x)")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Member C Implementation Tests")
    print("=" * 60)
    
    try:
        test_days_since()
        test_availability_multiplier()
        test_behavioral_score()
        test_composite_score()
        test_disqualifier_penalty()
        test_multiplier_impact()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
