"""
TalentRank AI — Global Configuration

Centralizes all paths, scoring weights, and runtime ceiling constants.
Import this module instead of hardcoding magic numbers in pipeline stages.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

CANDIDATES_PATH = DATA_DIR / "candidates.jsonl"
JD_PATH = DATA_DIR / "job_description.docx"
SUBMISSION_PATH = PROJECT_ROOT / "submission.csv"

# Precomputed artifact paths
CANDIDATE_EMBEDDINGS_PATH = ARTIFACTS_DIR / "candidate_embeddings.npy"
JD_EMBEDDINGS_PATH = ARTIFACTS_DIR / "jd_embeddings.npy"
FAISS_INDEX_PATH = ARTIFACTS_DIR / "candidates.faiss"

# ---------------------------------------------------------------------------
# Scoring weights (Stage 5 composite formula)
# ---------------------------------------------------------------------------
WEIGHT_SKILL_MATCH = 0.35
WEIGHT_CAREER_MATCH = 0.25
WEIGHT_BEHAVIORAL = 0.20
WEIGHT_AVAILABILITY_STABILITY = 0.10
WEIGHT_SENIORITY_SHIPPING = 0.10

# Disqualifier penalty — heavy, near-zeroing for explicit JD "do not want" hits
DISQUALIFIER_PENALTY_SCALE = 0.85

# Availability multiplier range (derived from redrob_signals)
AVAILABILITY_MULTIPLIER_MIN = 0.60
AVAILABILITY_MULTIPLIER_MAX = 1.15

# ---------------------------------------------------------------------------
# Ranking constraints
# ---------------------------------------------------------------------------
TOP_K = 100                        # Number of candidates in the final output
TOTAL_CANDIDATES = 100_000         # Expected dataset size

# ---------------------------------------------------------------------------
# Runtime ceilings (ranking step only)
# ---------------------------------------------------------------------------
MAX_RANKING_WALL_CLOCK_SECONDS = 300   # 5 minutes
MAX_RANKING_RAM_GB = 16                # 16 GB
GPU_ALLOWED = False
NETWORK_ALLOWED = False

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384  # bge-small-en-v1.5 output dimension

# ---------------------------------------------------------------------------
# FAISS retrieval
# ---------------------------------------------------------------------------
FAISS_NPROBE = 16          # Number of Voronoi cells to probe at query time
FAISS_TOP_K_RETRIEVAL = 500  # Retrieve this many before re-scoring

# ---------------------------------------------------------------------------
# Optional reranker (LightGBM stretch goal)
# ---------------------------------------------------------------------------
RERANKER_ENABLED = False
RERANKER_MODEL_PATH = ARTIFACTS_DIR / "reranker.lgbm"
