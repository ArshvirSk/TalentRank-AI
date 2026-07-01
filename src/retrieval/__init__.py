"""
Retrieval sub-package — Semantic Search & Retrieval.

Owner: Member 2

Stages covered:
  1 — JD Understanding
  3 — Candidate Embedding Generation
  4 — FAISS Semantic Retrieval

Note: Heavy imports (sentence_transformers, faiss) are lazy-loaded.
      Only JDProfile is imported eagerly since it's a lightweight dataclass.
"""

from src.retrieval.jd_parser import JDProfile

__all__ = [
    "JDProfile",
]
