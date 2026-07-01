"""
test_build_texts.py
Pytest test suite for Person B — build_texts.py and scoring sanity.

Run fast tests (no model download):
    pytest test_build_texts.py -m fast -v

Run all tests (requires BAAI/bge-small-en-v1.5 to be cached):
    pytest test_build_texts.py -v

Requirements covered:
    2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3, 1.5, 12.1, 13.1-13.6
"""
import copy
import json
import os

import pytest

from src.retrieval.build_texts import (
    build_candidate_text,
    _build_redrob_section,
    _assign_tier,
    _assign_tier_from_proficiency,
    JD_MUST_HAVE,
    JD_NICE_TO_HAVE,
    JD_DISQUALIFIER,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample_candidates.json")


def _load_sample():
    with open(SAMPLE_PATH, encoding="utf-8") as f:
        return json.load(f)


def _minimal_candidate(candidate_id="TEST_001", **overrides):
    """Return a minimal valid candidate dict for unit tests."""
    c = {
        "candidate_id": candidate_id,
        "profile": {
            "headline": "Test Engineer",
            "summary": "Test summary",
            "current_title": "Engineer",
            "current_company": "TestCo",
            "current_industry": "Software",
            "current_company_size": "51-200",
            "years_of_experience": 5.0,
        },
        "career_history": [],
        "skills": [],
        "certifications": [],
        "education": [],
        "redrob_signals": {},
    }
    c.update(overrides)
    return c


# ---------------------------------------------------------------------------
# Sanity smoke test (original behaviour, no model)
# ---------------------------------------------------------------------------

def test_basic_text_construction_runs():
    """Smoke: build_candidate_text does not raise on a minimal candidate."""
    c = _minimal_candidate()
    text = build_candidate_text(c)
    assert isinstance(text, str)
    assert len(text) > 0


def test_jd_word_counts():
    """JD query strings have a reasonable word count (sanity)."""
    assert len(JD_MUST_HAVE.split()) > 50
    assert len(JD_NICE_TO_HAVE.split()) > 30
    assert len(JD_DISQUALIFIER.split()) > 50


def test_sample_candidates_build_without_error():
    """All 50 sample candidates build without raising."""
    data = _load_sample()
    for c in data:
        text = build_candidate_text(c)
        assert isinstance(text, str), f"Expected str for {c['candidate_id']}"


# ---------------------------------------------------------------------------
# Task 7.3 / Requirements 2.1, 13.1 — github_activity_score = -1 sentinel
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_github_sentinel_minus_one():
    """When github_activity_score == -1, text contains 'no account' not '-1'.
    Requirements: 2.1, 13.1
    """
    c = _minimal_candidate()
    c["redrob_signals"] = {"github_activity_score": -1}
    text = build_candidate_text(c)
    print(f"\n[DEBUG] candidate text:\n{text}\n")
    assert "no account" in text, f"Expected 'no account' in text but got: {text[:300]}"
    # '-1' should not appear as a score value (it appears nowhere in redrob section)
    redrob_idx = text.find("GitHub")
    if redrob_idx != -1:
        redrob_snippet = text[redrob_idx:redrob_idx + 50]
        assert "-1" not in redrob_snippet, (
            f"Sentinel -1 leaked into text near GitHub phrase: {redrob_snippet}"
        )


@pytest.mark.fast
def test_github_score_non_sentinel():
    """When github_activity_score >= 0, numeric value appears in text.
    Requirements: 2.2
    """
    c = _minimal_candidate()
    c["redrob_signals"] = {"github_activity_score": 42}
    text = build_candidate_text(c)
    print(f"\n[DEBUG] candidate text:\n{text}\n")
    assert "GitHub activity score" in text
    assert "42" in text


# ---------------------------------------------------------------------------
# Task 7.3 / Requirements 2.3, 13.2 — offer_acceptance_rate = -1 sentinel
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_offer_rate_sentinel_minus_one():
    """When offer_acceptance_rate == -1, text contains 'none' not '-1'.
    Requirements: 2.3, 13.2
    """
    c = _minimal_candidate()
    c["redrob_signals"] = {"offer_acceptance_rate": -1}
    text = build_candidate_text(c)
    print(f"\n[DEBUG] candidate text:\n{text}\n")
    assert "none" in text.lower(), f"Expected 'none' in text but got: {text[:300]}"
    offer_idx = text.find("Offer")
    if offer_idx != -1:
        offer_snippet = text[offer_idx:offer_idx + 50]
        assert "-1" not in offer_snippet, (
            f"Sentinel -1 leaked into text near Offer phrase: {offer_snippet}"
        )


@pytest.mark.fast
def test_offer_rate_non_sentinel():
    """When offer_acceptance_rate >= 0, numeric value appears in text.
    Requirements: 2.4
    """
    c = _minimal_candidate()
    c["redrob_signals"] = {"offer_acceptance_rate": 0.75}
    text = build_candidate_text(c)
    print(f"\n[DEBUG] candidate text:\n{text}\n")
    assert "Offer acceptance rate" in text
    assert "0.75" in text


# ---------------------------------------------------------------------------
# Requirements 2.5 — missing redrob_signals key
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_missing_redrob_signals_no_error():
    """Candidate without 'redrob_signals' key builds without raising.
    Requirements: 2.5
    """
    c = _minimal_candidate()
    del c["redrob_signals"]
    text = build_candidate_text(c)  # must not raise
    assert isinstance(text, str)


# ---------------------------------------------------------------------------
# Requirements 3.1, 3.2, 3.3 — RedRob flags in text
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_open_to_work_flag_true():
    """open_to_work_flag=True → 'Actively seeking' in text. Requirements: 3.1"""
    c = _minimal_candidate()
    c["redrob_signals"] = {"open_to_work_flag": True}
    text = build_candidate_text(c)
    print(f"\n[DEBUG] candidate text:\n{text}\n")
    assert "Actively seeking" in text


@pytest.mark.fast
def test_willing_to_relocate_true():
    """willing_to_relocate=True → 'Willing to relocate' in text. Requirements: 3.2"""
    c = _minimal_candidate()
    c["redrob_signals"] = {"willing_to_relocate": True}
    text = build_candidate_text(c)
    print(f"\n[DEBUG] candidate text:\n{text}\n")
    assert "Willing to relocate" in text


@pytest.mark.fast
def test_profile_completeness_in_text():
    """profile_completeness_score appears as phrase. Requirements: 3.3"""
    c = _minimal_candidate()
    c["redrob_signals"] = {"profile_completeness_score": 87.5}
    text = build_candidate_text(c)
    print(f"\n[DEBUG] candidate text:\n{text}\n")
    assert "Profile completeness" in text
    assert "87.5" in text


# ---------------------------------------------------------------------------
# Skill trust fallback — Requirements 1.5, 12.1
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_skill_trust_path_none_falls_back_gracefully():
    """skill_trust_path=None uses raw proficiency fallback without error.
    Requirements: 1.5, 12.1
    """
    c = _minimal_candidate()
    c["skills"] = [
        {"name": "Python", "proficiency": "advanced", "duration_months": 48, "endorsements": 10},
        {"name": "SQL", "proficiency": "beginner", "duration_months": 6, "endorsements": 0},
    ]
    text = build_candidate_text(c, skill_trust_path=None)
    print(f"\n[DEBUG] candidate text:\n{text}\n")
    assert "Python" in text


@pytest.mark.fast
def test_beginner_skill_in_other_block():
    """A beginner-proficiency skill lands in 'Other skills' block, not 'Core skills'.
    Requirements: 1.4, 13.5
    """
    c = _minimal_candidate()
    c["skills"] = [
        {"name": "Photoshop", "proficiency": "beginner", "duration_months": 3, "endorsements": 0},
    ]
    text = build_candidate_text(c)
    print(f"\n[DEBUG] candidate text:\n{text}\n")
    # Either absent from Core/Skills blocks OR appears only in Other skills
    in_core = "Core skills:" in text and "Photoshop" in text.split("Core skills:")[1].split("|")[0]
    in_skills = "Skills:" in text and "Photoshop" in text.split("Skills:")[1].split("|")[0]
    if in_core or in_skills:
        pytest.fail(
            f"Beginner skill 'Photoshop' found in Core/Skills block. Full text:\n{text}"
        )


@pytest.mark.fast
def test_skill_deduplication():
    """Duplicate skill names appear only once in text. Requirements: 1.6"""
    c = _minimal_candidate()
    c["skills"] = [
        {"name": "Python", "proficiency": "advanced", "duration_months": 48, "endorsements": 20},
        {"name": "python", "proficiency": "expert", "duration_months": 60, "endorsements": 50},
    ]
    text = build_candidate_text(c)
    print(f"\n[DEBUG] candidate text:\n{text}\n")
    count = text.lower().count("python")
    assert count == 1, f"Expected 'python' once in text, found {count}. Text:\n{text}"


# ---------------------------------------------------------------------------
# Tier assignment helpers
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_assign_tier_buckets():
    """_assign_tier maps scores to correct tier strings."""
    assert _assign_tier(0.80) == "expert"
    assert _assign_tier(0.75) == "expert"
    assert _assign_tier(0.74) == "advanced"
    assert _assign_tier(0.50) == "advanced"
    assert _assign_tier(0.49) == "intermediate"
    assert _assign_tier(0.25) == "intermediate"
    assert _assign_tier(0.24) == "beginner"
    assert _assign_tier(0.00) == "beginner"


@pytest.mark.fast
def test_assign_tier_from_proficiency():
    """_assign_tier_from_proficiency is case-insensitive."""
    assert _assign_tier_from_proficiency("Expert") == "expert"
    assert _assign_tier_from_proficiency("ADVANCED") == "advanced"
    assert _assign_tier_from_proficiency("intermediate") == "intermediate"
    assert _assign_tier_from_proficiency("beginner") == "beginner"
    assert _assign_tier_from_proficiency("unknown") == "beginner"
    assert _assign_tier_from_proficiency("") == "beginner"


# ---------------------------------------------------------------------------
# Task 7.1 / Requirements 13.3 — relative sim_must ordering (needs model)
# ---------------------------------------------------------------------------

def _find_strong_weak_candidates(data):
    """
    Strong candidate: has the most career description text containing
    'embeddings', 'retrieval', 'vector', 'semantic search', or 'ranking'.
    Weak candidate: non-engineering title with no ML keywords in career.
    """
    ml_keywords = {"embeddings", "retrieval", "vector", "semantic", "ranking", "nlp", "bert"}
    non_eng_titles = {"operations manager", "marketing manager", "customer support",
                      "accountant", "business analyst", "hr manager", "content writer"}

    scored = []
    for c in data:
        descriptions = " ".join(
            role.get("description", "") for role in c.get("career_history", [])
        ).lower()
        
        # Base score based on ML keywords
        score = sum(descriptions.count(kw) for kw in ml_keywords)
        
        # Penalize strongly if they are a non-engineering title (to make them the weak candidate)
        curr_title = c.get("profile", {}).get("current_title", "").lower()
        is_non_eng = any(nt in curr_title for nt in non_eng_titles)
        if is_non_eng:
            score -= 1000
            
        scored.append((score, c))

    scored.sort(key=lambda x: x[0])
    weak = scored[0][1]   # lowest score (most disqualified)
    strong = scored[-1][1]  # highest score (most ML)
    return strong, weak


@pytest.mark.slow
def test_sim_must_strong_greater_than_weak():
    """Strong AI candidate must score higher on sim_must than weak candidate.
    Requirements: 13.3
    """
    from sentence_transformers import SentenceTransformer
    from src.retrieval.embed import MODEL_NAME, _prepare_query_texts
    import numpy as np

    data = _load_sample()
    strong, weak = _find_strong_weak_candidates(data)

    print(f"\n[DEBUG] Strong candidate: {strong['candidate_id']} — "
          f"{strong['profile']['current_title']} at {strong['profile']['current_company']}")
    print(f"[DEBUG] Weak candidate:   {weak['candidate_id']} — "
          f"{weak['profile']['current_title']} at {weak['profile']['current_company']}")

    model = SentenceTransformer(MODEL_NAME)
    query_texts = _prepare_query_texts([JD_MUST_HAVE, JD_NICE_TO_HAVE, JD_DISQUALIFIER], MODEL_NAME)
    qv = model.encode(query_texts, normalize_embeddings=True).astype("float32")

    strong_text = build_candidate_text(strong)
    weak_text = build_candidate_text(weak)

    strong_emb = model.encode([strong_text], normalize_embeddings=True).astype("float32")[0]
    weak_emb = model.encode([weak_text], normalize_embeddings=True).astype("float32")[0]

    sim_must_strong = float(np.dot(strong_emb, qv[0]))
    sim_must_weak = float(np.dot(weak_emb, qv[0]))

    print(f"[DEBUG] sim_must strong={sim_must_strong:.4f}  weak={sim_must_weak:.4f}")

    assert sim_must_strong > sim_must_weak, (
        f"Expected sim_must(strong) > sim_must(weak), "
        f"got {sim_must_strong:.4f} <= {sim_must_weak:.4f}\n"
        f"Strong text[:200]: {strong_text[:200]}\n"
        f"Weak text[:200]: {weak_text[:200]}"
    )


# ---------------------------------------------------------------------------
# Task 7.2 / Requirements 13.4 — sim_disq ordering (needs model)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_sim_disq_weak_greater_than_strong():
    """Weak/disqualifying candidate must score higher on sim_disq than strong candidate.
    Requirements: 13.4
    """
    from sentence_transformers import SentenceTransformer
    from src.retrieval.embed import MODEL_NAME, _prepare_query_texts
    import numpy as np

    data = _load_sample()
    strong, weak = _find_strong_weak_candidates(data)

    model = SentenceTransformer(MODEL_NAME)
    query_texts = _prepare_query_texts([JD_MUST_HAVE, JD_NICE_TO_HAVE, JD_DISQUALIFIER], MODEL_NAME)
    qv = model.encode(query_texts, normalize_embeddings=True).astype("float32")

    strong_text = build_candidate_text(strong)
    weak_text = build_candidate_text(weak)

    strong_emb = model.encode([strong_text], normalize_embeddings=True).astype("float32")[0]
    weak_emb = model.encode([weak_text], normalize_embeddings=True).astype("float32")[0]

    sim_disq_strong = float(np.dot(strong_emb, qv[2]))
    sim_disq_weak = float(np.dot(weak_emb, qv[2]))

    print(f"\n[DEBUG] sim_disq strong={sim_disq_strong:.4f}  weak={sim_disq_weak:.4f}")

    assert sim_disq_weak > sim_disq_strong, (
        f"Expected sim_disq(weak) > sim_disq(strong), "
        f"got {sim_disq_weak:.4f} <= {sim_disq_strong:.4f}\n"
        f"Strong: {strong['candidate_id']} — {strong['profile']['current_title']}\n"
        f"Weak:   {weak['candidate_id']} — {weak['profile']['current_title']}"
    )


# ---------------------------------------------------------------------------
# _build_redrob_section direct unit tests
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_redrob_section_empty_signals_returns_none():
    assert _build_redrob_section({}) is None
    assert _build_redrob_section(None) is None


@pytest.mark.fast
def test_redrob_section_all_false_returns_none():
    """Boolean False fields are silent — if that's all there is, return None."""
    result = _build_redrob_section({
        "open_to_work_flag": False,
        "willing_to_relocate": False,
    })
    assert result is None


@pytest.mark.fast
def test_redrob_section_assessment_scores():
    """Each skill assessment score produces one phrase."""
    result = _build_redrob_section({
        "skill_assessment_scores": {"NLP": 85.0, "Python": 92.3}
    })
    assert result is not None
    assert "NLP score: 85.0" in result
    assert "Python score: 92.3" in result


# ---------------------------------------------------------------------------
# Standalone sanity print (kept for backward compat with direct python execution)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data = _load_sample()
    print("=== TEXT CONSTRUCTION TEST ===\n")
    for c in data[:3]:
        text = build_candidate_text(c)
        print(f"ID: {c['candidate_id']} | Title: {c['profile']['current_title']}")
        print(f"Word count: {len(text.split())}")
        print(f"Preview: {text[:200]}...")
        print()

    print("=== JD VECTOR WORD COUNTS ===")
    print(f"MUST_HAVE:    {len(JD_MUST_HAVE.split())} words")
    print(f"NICE_TO_HAVE: {len(JD_NICE_TO_HAVE.split())} words")
    print(f"DISQUALIFIER: {len(JD_DISQUALIFIER.split())} words")
    print("\nAll look good — run `pytest test_build_texts.py -m fast` for full suite.")
