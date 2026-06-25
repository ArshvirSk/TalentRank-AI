# TalentRank AI

> Submission for the **Redrob Intelligent Candidate Discovery & Ranking Challenge**
> *(India Runs Data & AI Challenge)*

Rank the top 100 of 100,000 candidates against a fixed job description
(Senior AI Engineer — Redrob AI, Series A) using a fully offline, CPU-only
hybrid scoring pipeline.

---

## Quick Start

```bash
# 1. Clone & install
git clone <repo-url> && cd talentrank-ai
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt

# 2. Place data files
#    data/candidates.jsonl   — 100k candidate records
#    data/job_description.docx — the fixed JD

# 3. Precompute embeddings (one-time, no time limit)
python -m src.retrieval.embed --candidates ./data/candidates.jsonl --jd ./data/job_description.docx

# 4. Produce the final submission CSV (must finish in <=5 min)
python -m src.ranking.rank --candidates ./data/candidates.jsonl --jd ./data/job_description.docx --out ./submission.csv
```

---

## Architecture (8 Stages)

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: JD Understanding                                      │
│  Parse JD → must-have / nice-to-have / disqualifier vectors     │
├─────────────────────────────────────────────────────────────────┤
│  Stage 2: Structured Feature Extraction                         │
│  Per-candidate signals from career_history, skills, education   │
├─────────────────────────────────────────────────────────────────┤
│  Stage 3: Candidate Embedding Generation                        │
│  CPU-friendly sentence-transformers (offline precompute)        │
├─────────────────────────────────────────────────────────────────┤
│  Stage 4: FAISS Semantic Retrieval                              │
│  Three-vector query (must-have, nice-to-have, disqualifier)     │
├─────────────────────────────────────────────────────────────────┤
│  Stage 5: Hybrid Candidate Scoring                              │
│  Composite = weighted features + similarity − penalties         │
├─────────────────────────────────────────────────────────────────┤
│  Stage 6: Honeypot Detection Layer                              │
│  Rule-based consistency checks → hard exclusion gate             │
├─────────────────────────────────────────────────────────────────┤
│  Stage 7: Final Ranking Engine (rank.py CLI)                    │
│  Loads precomputed artifacts, applies 5+6, writes top-100 CSV   │
├─────────────────────────────────────────────────────────────────┤
│  Stage 8: Explanation Generator                                 │
│  Template-based, fact-only reasoning per candidate              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Compute Constraints (ranking step only)

| Constraint        | Limit                                                    |
|-------------------|----------------------------------------------------------|
| Wall-clock time   | **≤ 5 minutes**                                          |
| RAM               | **≤ 16 GB**                                              |
| Hardware          | **CPU-only** — no GPU                                    |
| Network           | **None** — fully offline                                 |
| Hosted LLMs       | **Prohibited** — no OpenAI / Anthropic / etc. API calls  |
| Precompute        | Embedding & model prep may run offline with no time limit |

---

## Team Ownership Map

| Member | Responsibility                         | Directory        | Stages |
|--------|----------------------------------------|------------------|--------|
| 1      | Data Engineering & Feature Pipeline    | `src/features/`  | 2, 6   |
| 2      | Semantic Search & Retrieval            | `src/retrieval/`  | 1, 3, 4|
| 3      | Ranking Engine & ML                    | `src/ranking/`    | 5, 7   |
| 4      | Frontend, Eval, Docs & Submission      | `src/explain/`, `app/`, `eval/`, `docs/` | 8 |

---

## Scoring Formula

```
base = 0.35 * skill_match
     + 0.25 * career_match
     + 0.20 * behavioral_score
     + 0.10 * availability_stability
     + 0.10 * seniority_and_shipping_score

base -= disqualifier_penalty          # heavy, near-zeroing

final = base * availability_multiplier  # ~0.6–1.15 from redrob_signals
```

---

## Output Format

```
candidate_id,rank,score,reasoning
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
TalentRankAI/
├── README.md
├── requirements.txt
├── Dockerfile
├── .gitignore
├── submission_metadata.yaml
├── config.py
├── data/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── schema.py
│   │   ├── career_signals.py
│   │   ├── skill_trust.py
│   │   └── honeypot.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── jd_parser.py
│   │   ├── embed.py
│   │   └── faiss_index.py
│   ├── ranking/
│   │   ├── __init__.py
│   │   ├── fusion.py
│   │   ├── behavioral.py
│   │   ├── reranker.py
│   │   └── rank.py
│   └── explain/
│       ├── __init__.py
│       ├── templates.py
│       └── generate.py
├── app/
│   └── streamlit_app.py
├── eval/
│   ├── spot_check.py
│   └── validate_submission.py
└── tests/
    ├── fixtures/
    │   └── sample_candidates.json
    ├── test_features.py
    ├── test_retrieval.py
    └── test_ranking.py
```
