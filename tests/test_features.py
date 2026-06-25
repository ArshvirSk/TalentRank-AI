"""
Tests for src.features (Member 1's modules).

Covers:
  - schema.py: CandidateRecord parsing
  - career_signals.py: signal extraction
  - skill_trust.py: trust-weighted skill scoring
  - honeypot.py: consistency-check detection
"""

import json
from pathlib import Path

import pytest

from src.features.schema import parse_candidate, CandidateRecord
from src.features.career_signals import extract_career_signals, CareerSignals
from src.features.skill_trust import compute_skill_trust_score
from src.features.honeypot import detect_honeypot, HoneypotResult


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_candidates() -> list[dict]:
    """Load the sample candidates fixture."""
    with open(FIXTURES_DIR / "sample_candidates.json", "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def good_candidate(sample_candidates) -> dict:
    """The strong candidate (TEST_001)."""
    return sample_candidates[0]


@pytest.fixture
def weak_candidate(sample_candidates) -> dict:
    """The weak candidate (TEST_002)."""
    return sample_candidates[1]


@pytest.fixture
def honeypot_candidate(sample_candidates) -> dict:
    """The honeypot candidate (HONEYPOT_001)."""
    return sample_candidates[2]


# ---- Schema parsing tests ----

class TestParseCandidate:
    """Tests for schema.parse_candidate."""

    @pytest.mark.skip(reason="Not yet implemented — TODO(Member 1)")
    def test_parse_good_candidate(self, good_candidate):
        """A well-formed candidate record should parse without errors."""
        record = parse_candidate(good_candidate)
        assert isinstance(record, CandidateRecord)
        assert record.candidate_id == "TEST_001"
        assert record.years_of_experience == 7.5
        assert len(record.skills) == 4
        assert len(record.career_history) == 2

    @pytest.mark.skip(reason="Not yet implemented — TODO(Member 1)")
    def test_parse_preserves_skill_details(self, good_candidate):
        """Skill entries should retain proficiency and endorsement data."""
        record = parse_candidate(good_candidate)
        python_skill = next(s for s in record.skills if s.name == "Python")
        assert python_skill.proficiency == "expert"
        assert python_skill.endorsements == 45


# ---- Career signals tests ----

class TestCareerSignals:
    """Tests for career_signals.extract_career_signals."""

    @pytest.mark.skip(reason="Not yet implemented — TODO(Member 1)")
    def test_production_evidence(self, good_candidate):
        """Candidate with production deployment keywords should be flagged."""
        record = parse_candidate(good_candidate)
        signals = extract_career_signals(record)
        assert signals.has_production_deployment is True

    @pytest.mark.skip(reason="Not yet implemented — TODO(Member 1)")
    def test_langchain_only_flag(self, weak_candidate):
        """Candidate with only LangChain and short career should be flagged."""
        record = parse_candidate(weak_candidate)
        signals = extract_career_signals(record)
        assert signals.is_langchain_only_recent is True


# ---- Skill trust tests ----

class TestSkillTrust:
    """Tests for skill_trust.compute_skill_trust_score."""

    @pytest.mark.skip(reason="Not yet implemented — TODO(Member 1)")
    def test_score_is_non_negative(self, good_candidate):
        """Trust score should always be non-negative."""
        record = parse_candidate(good_candidate)
        score = compute_skill_trust_score(record)
        assert score >= 0.0

    @pytest.mark.skip(reason="Not yet implemented — TODO(Member 1)")
    def test_strong_candidate_scores_higher(self, good_candidate, weak_candidate):
        """A more experienced candidate should score higher."""
        strong = parse_candidate(good_candidate)
        weak = parse_candidate(weak_candidate)
        assert compute_skill_trust_score(strong) > compute_skill_trust_score(weak)


# ---- Honeypot detection tests ----

class TestHoneypot:
    """Tests for honeypot.detect_honeypot."""

    @pytest.mark.skip(reason="Not yet implemented — TODO(Member 1)")
    def test_detects_honeypot(self, honeypot_candidate):
        """Candidate with expert skills at 0 months should be flagged."""
        record = parse_candidate(honeypot_candidate)
        result = detect_honeypot(record)
        assert result.is_honeypot is True
        assert len(result.reasons) > 0

    @pytest.mark.skip(reason="Not yet implemented — TODO(Member 1)")
    def test_good_candidate_not_flagged(self, good_candidate):
        """A legitimate candidate should not be flagged as a honeypot."""
        record = parse_candidate(good_candidate)
        result = detect_honeypot(record)
        assert result.is_honeypot is False
