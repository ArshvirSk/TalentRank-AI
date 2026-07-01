"""
Comprehensive test suite for Member C's implementations:
  - behavioral.py: compute_availability_multiplier, compute_behavioral_score
  - fusion.py: compute_composite_score, ScoreBreakdown
  - rank.py: run_ranking_pipeline, write_submission_csv

Test fixtures and edge cases covering all signal types and corner cases.
"""

import pytest
import tempfile
import csv
from datetime import datetime, timedelta
from pathlib import Path

from src.features.schema import CandidateRecord
from src.features.career_signals import CareerSignals
from src.ranking.behavioral import (
    compute_availability_multiplier,
    compute_behavioral_score,
    _days_since,
    _get_signal_value,
)
from src.ranking.fusion import (
    compute_composite_score,
    ScoreBreakdown,
    _extract_availability_stability,
    _extract_seniority_and_shipping,
)
from src.ranking.rank import write_submission_csv


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def candidate_perfect() -> CandidateRecord:
    """Candidate with all perfect signals."""
    return CandidateRecord(
        candidate_id="CAND_0000001",
        profile={
            "anonymized_name": "Alice Perfect",
            "headline": "Senior Software Engineer",
            "summary": "10+ years experience",
            "location": "Mumbai",
            "country": "India",
            "years_of_experience": 10,
            "current_title": "Senior Engineer",
            "current_company": "FAANG Corp",
            "current_company_size": "10001+",
            "current_industry": "Technology",
        },
        career_history=[
            {
                "company": "FAANG Corp",
                "title": "Senior Engineer",
                "start_date": "2022-01-01",
                "end_date": None,
                "duration_months": 24,
                "is_current": True,
                "industry": "Technology",
                "company_size": "10001+",
                "description": "Technical leadership",
            },
            {
                "company": "Tech Startup",
                "title": "Engineer",
                "start_date": "2018-01-01",
                "end_date": "2022-01-01",
                "duration_months": 48,
                "is_current": False,
                "industry": "Technology",
                "company_size": "11-50",
                "description": "Core engineer",
            },
        ],
        education=[
            {
                "institution": "IIT Mumbai",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "start_year": 2014,
                "end_year": 2018,
                "grade": "9.5",
                "tier": "tier_1",
            },
        ],
        skills=[
            {"name": "Python", "proficiency": "expert", "endorsements": 50, "duration_months": 120},
            {"name": "System Design", "proficiency": "expert", "endorsements": 40, "duration_months": 100},
        ],
        redrob_signals={
            "profile_completeness_score": 100,
            "signup_date": "2018-01-01",
            "last_active_date": (datetime.now() - timedelta(days=1)).isoformat(),
            "open_to_work_flag": True,
            "profile_views_received_30d": 50,
            "applications_submitted_30d": 10,
            "recruiter_response_rate": 0.95,
            "avg_response_time_hours": 1,
            "skill_assessment_scores": {"Python": 95, "System Design": 90},
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
        },
    )


@pytest.fixture
def candidate_poor() -> CandidateRecord:
    """Candidate with all poor signals."""
    return CandidateRecord(
        candidate_id="CAND_0000002",
        profile={
            "anonymized_name": "Bob Poor",
            "headline": "Entry-level Developer",
            "summary": "Junior developer",
            "location": "Remote",
            "country": "India",
            "years_of_experience": 0.5,
            "current_title": "Junior Dev",
            "current_company": "Startup",
            "current_company_size": "1-10",
            "current_industry": "Technology",
        },
        career_history=[
            {
                "company": "Startup",
                "title": "Junior Dev",
                "start_date": "2023-06-01",
                "end_date": None,
                "duration_months": 6,
                "is_current": True,
                "industry": "Technology",
                "company_size": "1-10",
                "description": "Junior role",
            },
        ],
        education=[],
        skills=[{"name": "JavaScript", "proficiency": "beginner", "endorsements": 0, "duration_months": 6}],
        redrob_signals={
            "profile_completeness_score": 30,
            "signup_date": "2023-06-01",
            "last_active_date": (datetime.now() - timedelta(days=180)).isoformat(),
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
        },
    )


@pytest.fixture
def career_signals_perfect() -> CareerSignals:
    """Perfect career signals."""
    return CareerSignals(
        years_of_experience=10.0,
        avg_tenure_months=48.0,
        title_progression_score=0.9,
        title_chaser_score=0.1,
        industry_consistency_score=0.95,
        company_quality_score=0.9,
        stability_score=0.95,
        seniority_score=0.95,
    )


@pytest.fixture
def career_signals_poor() -> CareerSignals:
    """Poor career signals."""
    return CareerSignals(
        years_of_experience=0.5,
        avg_tenure_months=6.0,
        title_progression_score=0.2,
        title_chaser_score=0.8,
        industry_consistency_score=0.3,
        company_quality_score=0.2,
        stability_score=0.2,
        seniority_score=0.1,
    )


# ============================================================================
# TESTS: _days_since utility
# ============================================================================

class TestDaysSince:
    """Test the _days_since utility function."""
    
    def test_recent_date(self):
        """Test with a recent date."""
        yesterday = (datetime.now() - timedelta(days=1)).isoformat()
        days = _days_since(yesterday)
        assert 0 <= days <= 2  # Allow 1-2 day discrepancy
    
    def test_old_date(self):
        """Test with an old date."""
        old_date = (datetime.now() - timedelta(days=1000)).isoformat()
        days = _days_since(old_date)
        assert 990 <= days <= 1010  # Allow small discrepancy
    
    def test_missing_date(self):
        """Test with missing date."""
        days = _days_since(None)
        assert days == 10000
    
    def test_empty_string(self):
        """Test with empty string."""
        days = _days_since("")
        assert days == 10000


# ============================================================================
# TESTS: compute_availability_multiplier
# ============================================================================

class TestAvailabilityMultiplier:
    """Test the compute_availability_multiplier function."""
    
    def test_perfect_candidate(self, candidate_perfect):
        """Test with perfect candidate signals."""
        multiplier = compute_availability_multiplier(candidate_perfect)
        assert 1.0 <= multiplier <= 1.15, f"Expected [1.0, 1.15], got {multiplier}"
    
    def test_poor_candidate(self, candidate_poor):
        """Test with poor candidate signals."""
        multiplier = compute_availability_multiplier(candidate_poor)
        assert 0.60 <= multiplier <= 0.85, f"Expected [0.60, 0.85], got {multiplier}"
    
    def test_multiplier_range(self, candidate_perfect):
        """Test that multiplier is always in valid range."""
        for _ in range(10):
            multiplier = compute_availability_multiplier(candidate_perfect)
            assert 0.60 <= multiplier <= 1.15, f"Out of range: {multiplier}"
    
    def test_notice_period_impact(self):
        """Test that short notice period increases multiplier."""
        candidate_short_notice = CandidateRecord(
            candidate_id="CAND_SHORT",
            profile={},
            career_history=[],
            education=[],
            skills=[],
            redrob_signals={
                "notice_period_days": 0,
                "open_to_work_flag": True,
                "willing_to_relocate": False,
                "recruiter_response_rate": 0.7,
                "interview_completion_rate": 0.7,
                "offer_acceptance_rate": -1,
                "last_active_date": datetime.now().isoformat(),
                "profile_completeness_score": 50,
                "verified_email": True,
                "verified_phone": False,
                "linkedin_connected": False,
                "github_activity_score": -1,
                "profile_views_received_30d": 10,
                "search_appearance_30d": 5,
                "saved_by_recruiters_30d": 2,
                "applications_submitted_30d": 3,
                "connection_count": 100,
                "preferred_work_mode": "hybrid",
                "avg_response_time_hours": 24,
                "skill_assessment_scores": {},
            },
        )
        
        candidate_long_notice = CandidateRecord(
            candidate_id="CAND_LONG",
            profile={},
            career_history=[],
            education=[],
            skills=[],
            redrob_signals={
                "notice_period_days": 120,  # Different notice period
                "open_to_work_flag": True,
                "willing_to_relocate": False,
                "recruiter_response_rate": 0.7,
                "interview_completion_rate": 0.7,
                "offer_acceptance_rate": -1,
                "last_active_date": datetime.now().isoformat(),
                "profile_completeness_score": 50,
                "verified_email": True,
                "verified_phone": False,
                "linkedin_connected": False,
                "github_activity_score": -1,
                "profile_views_received_30d": 10,
                "search_appearance_30d": 5,
                "saved_by_recruiters_30d": 2,
                "applications_submitted_30d": 3,
                "connection_count": 100,
                "preferred_work_mode": "hybrid",
                "avg_response_time_hours": 24,
                "skill_assessment_scores": {},
            },
        )
        
        mult_short = compute_availability_multiplier(candidate_short_notice)
        mult_long = compute_availability_multiplier(candidate_long_notice)
        
        # Short notice should have higher multiplier
        assert mult_short > mult_long


# ============================================================================
# TESTS: compute_behavioral_score
# ============================================================================

class TestBehavioralScore:
    """Test the compute_behavioral_score function."""
    
    def test_perfect_candidate(self, candidate_perfect):
        """Test with perfect candidate signals."""
        score = compute_behavioral_score(candidate_perfect)
        assert 0.7 <= score <= 1.0, f"Expected [0.7, 1.0], got {score}"
    
    def test_poor_candidate(self, candidate_poor):
        """Test with poor candidate signals."""
        score = compute_behavioral_score(candidate_poor)
        assert 0.0 <= score <= 0.4, f"Expected [0.0, 0.4], got {score}"
    
    def test_score_range(self, candidate_perfect):
        """Test that score is always in [0, 1]."""
        for _ in range(10):
            score = compute_behavioral_score(candidate_perfect)
            assert 0.0 <= score <= 1.0, f"Out of range: {score}"
    
    def test_recency_impact(self):
        """Test that recent activity increases behavioral score."""
        candidate_active = CandidateRecord(
            candidate_id="CAND_ACTIVE",
            profile={},
            career_history=[],
            education=[],
            skills=[],
            redrob_signals={
                "profile_completeness_score": 50,
                "last_active_date": datetime.now().isoformat(),  # Very recent
                "interview_completion_rate": 0.8,
                "offer_acceptance_rate": 0.7,
                "skill_assessment_scores": {"Python": 80},
                "applications_submitted_30d": 5,
                "search_appearance_30d": 10,
                "recruiter_response_rate": 0.7,
            },
        )
        
        candidate_stale = CandidateRecord(
            candidate_id="CAND_STALE",
            profile={},
            career_history=[],
            education=[],
            skills=[],
            redrob_signals={
                "profile_completeness_score": 50,
                "last_active_date": (datetime.now() - timedelta(days=200)).isoformat(),  # Very stale
                "interview_completion_rate": 0.8,
                "offer_acceptance_rate": 0.7,
                "skill_assessment_scores": {"Python": 80},
                "applications_submitted_30d": 5,
                "search_appearance_30d": 10,
                "recruiter_response_rate": 0.7,
            },
        )
        
        score_active = compute_behavioral_score(candidate_active)
        score_stale = compute_behavioral_score(candidate_stale)
        
        # Active candidate should have higher score
        assert score_active > score_stale


# ============================================================================
# TESTS: ScoreBreakdown and compute_composite_score
# ============================================================================

class TestCompositeScore:
    """Test the composite scoring function."""
    
    def test_perfect_score_breakdown(self, career_signals_perfect):
        """Test composite score with perfect inputs."""
        breakdown = compute_composite_score(
            candidate_id="CAND_0000001",
            skill_match_sim=0.95,
            career_match_sim=0.90,
            disqualifier_sim=0.05,
            career_signals=career_signals_perfect,
            availability_multiplier=1.15,
            behavioral_score=0.95,
        )
        
        assert isinstance(breakdown, ScoreBreakdown)
        assert 0.8 <= breakdown.final_score <= 1.0, f"Unexpected score: {breakdown.final_score}"
    
    def test_poor_score_breakdown(self, career_signals_poor):
        """Test composite score with poor inputs."""
        breakdown = compute_composite_score(
            candidate_id="CAND_0000002",
            skill_match_sim=0.2,
            career_match_sim=0.15,
            disqualifier_sim=0.85,
            career_signals=career_signals_poor,
            availability_multiplier=0.60,
            behavioral_score=0.2,
        )
        
        assert isinstance(breakdown, ScoreBreakdown)
        assert breakdown.final_score <= 0.2, f"Should be low: {breakdown.final_score}"
    
    def test_disqualifier_penalty(self, career_signals_perfect):
        """Test that high disqualifier applies heavy penalty."""
        breakdown_no_disqualifier = compute_composite_score(
            candidate_id="CAND_1",
            skill_match_sim=0.7,
            career_match_sim=0.6,
            disqualifier_sim=0.1,
            career_signals=career_signals_perfect,
            availability_multiplier=1.0,
            behavioral_score=0.7,
        )
        
        breakdown_high_disqualifier = compute_composite_score(
            candidate_id="CAND_2",
            skill_match_sim=0.7,
            career_match_sim=0.6,
            disqualifier_sim=0.8,  # High disqualifier
            career_signals=career_signals_perfect,
            availability_multiplier=1.0,
            behavioral_score=0.7,
        )
        
        # High disqualifier should result in much lower score
        assert breakdown_high_disqualifier.final_score < breakdown_no_disqualifier.final_score
    
    def test_availability_multiplier_impact(self, career_signals_perfect):
        """Test that multiplier correctly scales the score."""
        breakdown_low_mult = compute_composite_score(
            candidate_id="CAND_1",
            skill_match_sim=0.7,
            career_match_sim=0.6,
            disqualifier_sim=0.1,
            career_signals=career_signals_perfect,
            availability_multiplier=0.60,
            behavioral_score=0.7,
        )
        
        breakdown_high_mult = compute_composite_score(
            candidate_id="CAND_2",
            skill_match_sim=0.7,
            career_match_sim=0.6,
            disqualifier_sim=0.1,
            career_signals=career_signals_perfect,
            availability_multiplier=1.15,
            behavioral_score=0.7,
        )
        
        # High multiplier should result in higher score
        ratio = breakdown_high_mult.final_score / breakdown_low_mult.final_score
        assert 1.5 <= ratio <= 2.0, f"Multiplier effect should be ~1.91x, got {ratio}"
    
    def test_score_range(self, career_signals_perfect):
        """Test that final score is always in [0, 1]."""
        for skill in [0.1, 0.5, 0.9]:
            for disq in [0.1, 0.5, 0.9]:
                breakdown = compute_composite_score(
                    candidate_id="CAND_TEST",
                    skill_match_sim=skill,
                    career_match_sim=skill,
                    disqualifier_sim=disq,
                    career_signals=career_signals_perfect,
                    availability_multiplier=1.0,
                    behavioral_score=0.7,
                )
                assert 0.0 <= breakdown.final_score <= 1.0, f"Out of range: {breakdown.final_score}"


# ============================================================================
# TESTS: write_submission_csv
# ============================================================================

class TestWriteSubmissionCSV:
    """Test the CSV writing function."""
    
    def test_csv_format(self):
        """Test that CSV is written in correct format."""
        rows = [
            ("CAND_0000001", 1, 0.95, "High skill match"),
            ("CAND_0000002", 2, 0.92, "Good experience"),
            ("CAND_0000003", 3, 0.88, "Average fit"),
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_submission.csv"
            write_submission_csv(csv_path, rows)
            
            # Read back and verify
            assert csv_path.exists()
            
            with open(csv_path, 'r', newline='') as f:
                reader = csv.reader(f)
                lines = list(reader)
            
            # Check header
            assert lines[0] == ["candidate_id", "rank", "score", "reasoning"]
            
            # Check data rows
            assert len(lines) == 4  # header + 3 data rows
            assert lines[1][0] == "CAND_0000001"
            assert lines[1][1] == "1"
            assert float(lines[1][2]) == 0.95
    
    def test_csv_empty_rows(self):
        """Test CSV with no data rows."""
        rows = []
        
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_empty.csv"
            write_submission_csv(csv_path, rows)
            
            with open(csv_path, 'r', newline='') as f:
                reader = csv.reader(f)
                lines = list(reader)
            
            # Should have just header
            assert len(lines) == 1
            assert lines[0] == ["candidate_id", "rank", "score", "reasoning"]
    
    def test_csv_special_characters(self):
        """Test CSV with special characters in reasoning."""
        rows = [
            ("CAND_0000001", 1, 0.95, 'Reasoning with "quotes" and, commas'),
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_special.csv"
            write_submission_csv(csv_path, rows)
            
            with open(csv_path, 'r', newline='') as f:
                reader = csv.reader(f)
                lines = list(reader)
            
            assert len(lines) == 2
            assert lines[1][3] == 'Reasoning with "quotes" and, commas'


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_scoring_pipeline(self, candidate_perfect, career_signals_perfect):
        """Test the full scoring pipeline from candidate to final score."""
        # Compute availability multiplier
        multiplier = compute_availability_multiplier(candidate_perfect)
        assert 1.0 <= multiplier <= 1.15
        
        # Compute behavioral score
        behavioral = compute_behavioral_score(candidate_perfect)
        assert 0.7 <= behavioral <= 1.0
        
        # Compute composite score
        breakdown = compute_composite_score(
            candidate_id=candidate_perfect.candidate_id,
            skill_match_sim=0.9,
            career_match_sim=0.85,
            disqualifier_sim=0.1,
            career_signals=career_signals_perfect,
            availability_multiplier=multiplier,
            behavioral_score=behavioral,
        )
        
        assert isinstance(breakdown, ScoreBreakdown)
        assert 0.5 <= breakdown.final_score <= 1.0
    
    def test_two_candidates_ranking(self, candidate_perfect, candidate_poor):
        """Test that perfect candidate scores higher than poor candidate."""
        career_perfect = CareerSignals(
            years_of_experience=10.0, avg_tenure_months=48.0, title_progression_score=0.9,
            title_chaser_score=0.1, industry_consistency_score=0.95, company_quality_score=0.9,
            stability_score=0.95, seniority_score=0.95,
        )
        career_poor = CareerSignals(
            years_of_experience=0.5, avg_tenure_months=6.0, title_progression_score=0.2,
            title_chaser_score=0.8, industry_consistency_score=0.3, company_quality_score=0.2,
            stability_score=0.2, seniority_score=0.1,
        )
        
        mult_perfect = compute_availability_multiplier(candidate_perfect)
        mult_poor = compute_availability_multiplier(candidate_poor)
        
        behavioral_perfect = compute_behavioral_score(candidate_perfect)
        behavioral_poor = compute_behavioral_score(candidate_poor)
        
        score_perfect = compute_composite_score(
            candidate_id=candidate_perfect.candidate_id,
            skill_match_sim=0.9, career_match_sim=0.85, disqualifier_sim=0.05,
            career_signals=career_perfect, availability_multiplier=mult_perfect,
            behavioral_score=behavioral_perfect,
        ).final_score
        
        score_poor = compute_composite_score(
            candidate_id=candidate_poor.candidate_id,
            skill_match_sim=0.2, career_match_sim=0.15, disqualifier_sim=0.7,
            career_signals=career_poor, availability_multiplier=mult_poor,
            behavioral_score=behavioral_poor,
        ).final_score
        
        assert score_perfect > score_poor


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
