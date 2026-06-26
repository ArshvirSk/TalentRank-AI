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
PROCESSED_DIR = DATA_DIR / "processed"
CANDIDATE_FEATURES_PATH = PROCESSED_DIR / "candidate_features.parquet"
CANDIDATE_SKILL_TRUST_PATH = PROCESSED_DIR / "candidate_skill_trust.parquet"

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

# ===========================================================================
# MEMBER A CONFIGURATION — Feature Pipeline
# ===========================================================================

# ---------------------------------------------------------------------------
# Industry → employer_type classification map
# Maps lowercase substrings found in the `industry` field to a type.
# Order matters: first match wins, so put more specific patterns first.
# ---------------------------------------------------------------------------
INDUSTRY_TO_EMPLOYER_TYPE: dict[str, str] = {
    # ---- services ----
    "it services": "services",
    "information technology & services": "services",
    "information technology and services": "services",
    "consulting": "services",
    "management consulting": "services",
    "bpo": "services",
    "business process outsourcing": "services",
    "staffing": "services",
    "staffing and recruiting": "services",
    "outsourcing": "services",
    "professional services": "services",
    "it consulting": "services",
    "technology consulting": "services",
    "human resources": "services",
    "recruitment": "services",
    # ---- product ----
    "software": "product",
    "software development": "product",
    "computer software": "product",
    "internet": "product",
    "saas": "product",
    "artificial intelligence": "product",
    "ai": "product",
    "machine learning": "product",
    "fintech": "product",
    "financial technology": "product",
    "e-commerce": "product",
    "ecommerce": "product",
    "gaming": "product",
    "computer games": "product",
    "telecommunications": "product",
    "semiconductors": "product",
    "consumer electronics": "product",
    "information technology": "product",      # generic IT = product (not services)
    "computer hardware": "product",
    "data analytics": "product",
    "cloud computing": "product",
    "cybersecurity": "product",
    "blockchain": "product",
    "edtech": "product",
    "healthtech": "product",
    "health tech": "product",
    "media": "product",
    "entertainment": "product",
    # ---- research ----
    "research": "research",
    "academia": "research",
    "higher education": "research",
    "education": "research",
    "think tank": "research",
    "r&d": "research",
}

# Default employer type when industry doesn't match any key above
EMPLOYER_TYPE_DEFAULT = "other"

# ---------------------------------------------------------------------------
# Production evidence keywords
# Scanned in career_history.description (case-insensitive)
# ---------------------------------------------------------------------------
PRODUCTION_KEYWORDS: list[str] = [
    "production", "deployed", "deploy", "deployment", "shipped", "ship",
    "launched", "launch", "scaled to", "scale to", "scaling",
    "users", "million users", "daily active", "dau", "mau",
    "latency", "uptime", "sla", "on-call", "oncall",
    "ci/cd", "cicd", "pipeline", "kubernetes", "k8s", "docker",
    "aws", "gcp", "azure", "cloud", "microservice",
    "api", "rest api", "real-time", "realtime",
    "monitoring", "observability", "grafana", "datadog",
    "load balancer", "cdn", "redis", "kafka",
    "served", "serving", "inference", "model serving",
]

# Recency decay: weight for current role vs older roles
PRODUCTION_RECENCY_WEIGHT_CURRENT = 1.0
PRODUCTION_RECENCY_DECAY_PER_POSITION = 0.15   # decay for each older position

# ---------------------------------------------------------------------------
# Title-chaser detection
# ---------------------------------------------------------------------------
SENIORITY_TITLE_KEYWORDS: list[str] = [
    "chief", "cto", "ceo", "coo", "cio", "vp", "vice president",
    "director", "head of", "head", "lead", "principal", "senior",
    "staff", "architect", "manager", "founding",
]

TITLE_CHASER_MIN_EMPLOYERS = 3
TITLE_CHASER_MAX_AVG_TENURE_MONTHS = 18

# ---------------------------------------------------------------------------
# Research-only detection
# ---------------------------------------------------------------------------
RESEARCH_KEYWORDS: list[str] = [
    "research", "researcher", "publication", "published", "paper",
    "thesis", "phd", "postdoc", "post-doc", "professor",
    "journal", "conference paper", "arxiv", "ieee", "acm",
    "academic", "lab", "laboratory",
]

# ---------------------------------------------------------------------------
# Recent-LLM-only detection
# ---------------------------------------------------------------------------
LLM_GENAI_SKILL_KEYWORDS: list[str] = [
    "langchain", "openai", "gpt", "chatgpt", "llm", "large language model",
    "fine-tuning", "fine tuning", "finetuning",
    "rag", "retrieval augmented", "retrieval-augmented",
    "prompt engineering", "prompt design",
    "generative ai", "genai", "gen ai",
    "llama", "mistral", "claude", "gemini",
    "hugging face", "huggingface", "transformers",
    "vector database", "vector db", "pinecone", "weaviate", "chroma",
]

# Skills that indicate genuinely deep ML background (not just LLM wrappers)
DEEP_ML_SKILL_KEYWORDS: list[str] = [
    "machine learning", "deep learning", "neural network",
    "nlp", "natural language processing",
    "computer vision", "cv", "image recognition",
    "mlops", "ml ops", "model training",
    "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn",
    "reinforcement learning", "rl",
    "recommendation system", "recommender",
    "feature engineering", "data science",
    "xgboost", "lightgbm", "gradient boosting",
    "bayesian", "statistical modeling",
    "speech recognition", "asr",
]

RECENT_LLM_ONLY_MAX_DURATION_MONTHS = 12
DEEP_ML_MIN_DURATION_MONTHS = 24

# ---------------------------------------------------------------------------
# CV / Speech / Robotics without NLP detection
# ---------------------------------------------------------------------------
CV_SPEECH_ROBOTICS_KEYWORDS: list[str] = [
    "computer vision", "image recognition", "image classification",
    "object detection", "yolo", "opencv", "image segmentation",
    "speech recognition", "speech synthesis", "text to speech", "tts",
    "asr", "automatic speech recognition",
    "robotics", "ros", "robot operating system",
    "autonomous driving", "self-driving", "lidar", "slam",
    "3d reconstruction", "point cloud",
]

NLP_IR_LLM_KEYWORDS: list[str] = [
    "nlp", "natural language processing", "natural language understanding",
    "text mining", "text classification", "text generation",
    "information retrieval", "search engine", "elasticsearch",
    "sentiment analysis", "named entity", "ner",
    "question answering", "qa",
    "chatbot", "conversational ai", "dialogue",
    "language model", "llm", "gpt", "bert", "transformer",
    "langchain", "rag", "prompt engineering",
    "machine translation", "mt",
    "summarization", "text summarization",
]

# ---------------------------------------------------------------------------
# Stale-coder detection
# ---------------------------------------------------------------------------
MANAGEMENT_TITLE_KEYWORDS: list[str] = [
    "manager", "director", "vp", "vice president",
    "head of", "head", "chief", "cto", "ceo", "coo", "cio",
    "president", "svp", "evp", "avp",
]

HANDS_ON_ENGINEERING_KEYWORDS: list[str] = [
    "code", "coding", "develop", "programming", "engineered",
    "implemented", "built", "designed", "architected",
    "python", "java", "javascript", "typescript", "c++", "rust", "golang",
    "sql", "database", "api", "backend", "frontend",
    "deploy", "kubernetes", "docker", "ci/cd", "git",
    "debug", "refactor", "prototype", "algorithm",
    "machine learning", "deep learning", "model", "training",
    "data pipeline", "etl", "spark",
]

STALE_CODER_MIN_EXPERIENCE_YEARS = 6

# ---------------------------------------------------------------------------
# Skill trust scoring
# ---------------------------------------------------------------------------
PROFICIENCY_WEIGHTS: dict[str, float] = {
    "beginner": 0.15,
    "intermediate": 0.40,
    "advanced": 0.70,
    "expert": 1.00,
}

# If endorsements==0, duration_months<3, AND mentioned_in_text is False,
# multiply the raw trust score by this discount factor.
SKILL_PADDING_DISCOUNT = 0.05

# Duration months normalization cap (diminishing returns beyond this)
SKILL_DURATION_CAP_MONTHS = 120

# Endorsement normalization cap
SKILL_ENDORSEMENT_CAP = 50

# ---------------------------------------------------------------------------
# Honeypot detection
# ---------------------------------------------------------------------------
HONEYPOT_EXPERT_MIN_DURATION_MONTHS = 6    # expert with < this → suspicious
HONEYPOT_EXPERIENCE_TOLERANCE_MONTHS = 18  # gap between reported & summed
HONEYPOT_OVERLAP_TOLERANCE_MONTHS = 3      # career overlap tolerance
HONEYPOT_FLAG_THRESHOLD = 0.3              # honeypot_score above this → flagged

# Profile completeness contradiction: high completeness but no verifications
HONEYPOT_COMPLETENESS_THRESHOLD = 95

