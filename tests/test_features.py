import pytest
from pathlib import Path

from src.features.schema import load_candidates
from src.features.career_signals import extract_career_signals
from src.features.skill_trust import get_skill_trust_table
from src.features.honeypot import compute_honeypot_score

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_candidates.jsonl"


def test_schema_loader():
    candidates = list(load_candidates(FIXTURE_PATH))
    assert len(candidates) == 1
    
    cand = candidates[0]
    assert cand.candidate_id == "CAND_0000005"
    assert cand.years_of_experience == 11.0
    assert len(cand.career_history) == 4
    assert len(cand.skills) == 6
    assert cand.redrob_signals.profile_completeness_score == 84.6


def test_career_signals():
    candidates = list(load_candidates(FIXTURE_PATH))
    cand = candidates[0]
    
    signals = extract_career_signals(cand)
    assert signals.candidate_id == "CAND_0000005"
    assert signals.employer_type_current == "other"  # Manufacturing is unmapped, so "other"
    assert isinstance(signals.employer_type_current, str)
    assert not signals.recent_llm_only_flag
    assert not signals.consulting_only_flag


def test_skill_trust():
    candidates = list(load_candidates(FIXTURE_PATH))
    cand = candidates[0]
    
    table = get_skill_trust_table(cand)
    assert len(table) == 6
    for row in table:
        assert "candidate_id" in row
        assert "skill_name" in row
        assert "trust_score" in row
        assert 0.0 <= row["trust_score"] <= 1.0


def test_honeypot_score():
    candidates = list(load_candidates(FIXTURE_PATH))
    cand = candidates[0]
    
    score, flag = compute_honeypot_score(cand)
    assert 0.0 <= score <= 1.0
    assert isinstance(flag, bool)
