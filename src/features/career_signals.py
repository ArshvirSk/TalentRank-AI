"""
Stage 2 — Structured career-signal extraction.

Owner: Member 1

Extracts per-candidate boolean/numeric signals from career_history, skills,
and education that feed into Stage 5 scoring:

  - production-deployment evidence
  - product-vs-services/consulting classification
  - title-chaser detector (many short stints with escalating titles)
  - research-only flag (academia/papers with no industry shipping)
  - recent-LangChain-only flag (hype-cycle-only skills)
  - CV/speech/robotics-without-NLP flag (adjacent-but-wrong AI niche)
"""

from __future__ import annotations

from dataclasses import dataclass
from src.features.schema import CandidateRecord


@dataclass
class CareerSignals:
    """
    Container for all extracted career signals for one candidate.

    Each field is a float in [0, 1] or a boolean flag.  Downstream stages
    (fusion.py, behavioral.py) consume these without re-parsing raw records.
    """
    has_production_deployment: bool = False
    product_company_ratio: float = 0.0          # fraction of career at product cos
    is_title_chaser: bool = False
    is_research_only: bool = False
    is_langchain_only_recent: bool = False
    is_cv_speech_robotics_no_nlp: bool = False
    total_career_months: int = 0
    seniority_score: float = 0.0                # 0–1 normalized
    shipping_score: float = 0.0                 # 0–1, evidence of launching products

    # TODO(Member 1): add any additional derived signals needed by fusion.py


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

    Raises
    ------
    NotImplementedError
        Scaffold — not yet implemented.

    Notes
    -----
    Implementation checklist (Member 1):
      1. Scan career_history descriptions for production/deployment keywords.
      2. Classify each company as product vs services/consulting.
      3. Detect title-chasing: >=3 positions < 12 months each with
         strictly increasing seniority titles.
      4. Research-only: all positions at universities/labs AND no shipping
         keywords.
      5. LangChain-only-recent: if "langchain" appears in skills but no other
         substantial ML/AI framework AND career < 2 years.
      6. CV/Speech/Robotics-without-NLP: domain mismatch flag.
    """
    # TODO(Member 1): implement signal extraction logic
    raise NotImplementedError(
        "extract_career_signals: signal extraction not yet implemented"
    )
