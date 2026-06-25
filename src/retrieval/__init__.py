"""
Retrieval sub-package — Semantic Search & Retrieval.

Owner: Member 2

Stages covered:
  1 — JD Understanding
  3 — Candidate Embedding Generation
  4 — FAISS Semantic Retrieval
"""

from src.retrieval.jd_parser import parse_job_description, JDProfile
from src.retrieval.embed import embed_texts, precompute_all_embeddings
from src.retrieval.faiss_index import build_faiss_index, query_faiss_index

__all__ = [
    "parse_job_description",
    "JDProfile",
    "embed_texts",
    "precompute_all_embeddings",
    "build_faiss_index",
    "query_faiss_index",
]
