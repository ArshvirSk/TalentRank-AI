"""
Tests for src.ranking (Member 3's modules).

Covers:
  - fusion.py: composite scoring
  - behavioral.py: availability multiplier
  - rank.py: CSV output format
"""

import csv
from pathlib import Path

import pytest

from src.ranking.fusion import compute_composite_score, ScoreBreakdown
from src.ranking.behavioral import compute_availability_multiplier
from src.ranking.rank import write_submission_csv
from src.features.career_signals import CareerSignals

import config


# ---- Fusion scoring tests ----

class TestFusion:
    """Tests for fusion.compute_composite_score."""

    @pytest.mark.skip(reason="Not yet implemented — TODO(Member 3)")
    def test_score_in_unit_range(self):
        """Final score should be clamped to [0, 1]."""
        signals = CareerSignals(
            has_production_deployment=True,
            product_company_ratio=0.8,
            seniority_score=0.7,
            shipping_score=0.6,
        )
        breakdown = compute_composite_score(
            candidate_id="TEST_001",
            skill_match_sim=0.85,
            career_match_sim=0.75,
            disqualifier_sim=0.1,
            career_signals=signals,
            availability_multiplier=1.0,
        )
        assert 0.0 <= breakdown.final_score <= 1.0

    @pytest.mark.skip(reason="Not yet implemented — TODO(Member 3)")
    def test_disqualifier_penalty_reduces_score(self):
        """High disqualifier similarity should heavily penalize the score."""
        signals = CareerSignals()
        low_disq = compute_composite_score(
            candidate_id="A", skill_match_sim=0.8, career_match_sim=0.7,
            disqualifier_sim=0.1, career_signals=signals, availability_multiplier=1.0,
        )
        high_disq = compute_composite_score(
            candidate_id="A", skill_match_sim=0.8, career_match_sim=0.7,
            disqualifier_sim=0.9, career_signals=signals, availability_multiplier=1.0,
        )
        assert high_disq.final_score < low_disq.final_score

    @pytest.mark.skip(reason="Not yet implemented — TODO(Member 3)")
    def test_availability_multiplier_effect(self):
        """Multiplier < 1 should reduce score, > 1 should increase it."""
        signals = CareerSignals()
        base_args = dict(
            candidate_id="A", skill_match_sim=0.8, career_match_sim=0.7,
            disqualifier_sim=0.1, career_signals=signals,
        )
        low = compute_composite_score(**base_args, availability_multiplier=0.6)
        high = compute_composite_score(**base_args, availability_multiplier=1.15)
        assert low.final_score < high.final_score


# ---- Behavioral tests ----

class TestBehavioral:
    """Tests for behavioral.compute_availability_multiplier."""

    @pytest.mark.skip(reason="Not yet implemented — TODO(Member 3)")
    def test_multiplier_in_range(self, sample_candidates):
        """Multiplier should be within configured bounds."""
        from src.features.schema import parse_candidate
        record = parse_candidate(sample_candidates[0])
        mult = compute_availability_multiplier(record)
        assert config.AVAILABILITY_MULTIPLIER_MIN <= mult <= config.AVAILABILITY_MULTIPLIER_MAX


# ---- CSV output tests ----

class TestCSVOutput:
    """Tests for rank.write_submission_csv."""

    def test_write_csv_format(self, tmp_path):
        """Output CSV should have correct header and row count."""
        out_path = tmp_path / "test_submission.csv"
        rows = [
            ("CAND_001", 1, 0.95, "Strong match on all dimensions."),
            ("CAND_002", 2, 0.90, "Good technical fit."),
        ]
        write_submission_csv(out_path, rows)

        with open(out_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ["candidate_id", "rank", "score", "reasoning"]
            data_rows = list(reader)
            assert len(data_rows) == 2
            assert data_rows[0][0] == "CAND_001"
            assert data_rows[0][1] == "1"
