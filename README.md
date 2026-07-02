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

# 3. Process the Data (Member A's Scope)
# This will read the 100k JSONL records and compress them into highly optimized Parquet files
python scripts/build_features.py

# 4. Precompute embeddings (one-time, no time limit)
python -m src.retrieval.embed --candidates ./data/candidates.jsonl --jd ./data/job_description.docx

# 5. Produce the final submission CSV (must finish in <=5 min)
python -m src.ranking.rank --candidates ./data/candidates.jsonl --jd ./data/job_description.docx --out ./submission.csv

# 6. Validate the Submission CSV
python -m eval.validate_submission .\submission.csv

# 7. Run Face Validity Spot Checks
python -m eval.spot_check

# 8. Launch Interactive Streamlit Dashboard
streamlit run app/streamlit_app.py
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

## Team — ASK

| Member | Name                    | Responsibility                         | Directory        | Stages |
|--------|-------------------------|----------------------------------------|------------------|--------|
| A      | **Arshvir Singh Kalsi** | Data Engineering & Feature Pipeline    | `src/features/`  | 2, 6   |
| B      | **Siddhant Sawant**     | Semantic Search & Retrieval            | `src/retrieval/`  | 1, 3, 4|
| C      | **Viraj Prabhu**        | Ranking Engine & ML Scoring            | `src/ranking/`    | 5, 7   |
| D      | **Ghrani Ganesh Poojari** | Frontend, Eval, Docs & Submission    | `app/`, `eval/`   | 8      |

> See [CONTRIBUTORS.md](CONTRIBUTORS.md) for detailed per-member contribution breakdown.

---

## Git Workflow & Collaboration

To prevent merge conflicts (especially on shared files like `config.py`) and ensure the pipeline stays unbroken, all team members should follow this workflow:

1. **Branching:** Do not push directly to `main`. Create a feature branch for your scope:
   ```bash
   git checkout -b feature/member-b-retrieval
   ```
2. **Pull Requests (PRs):** When finished with a stage, push your branch and open a Pull Request against `main`. 
3. **Review & Merge:** Have at least one other member review the PR (ensure no massive data files/Faiss indices were accidentally committed) before merging.
4. **Syncing:** Pull `main` frequently. Whenever a PR is merged, run `python scripts/build_features.py` (or the respective downstream script) to regenerate the latest artifacts locally.

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

## Data Schema Validation

The system strictly enforces data correctness using `src.features.schema.CandidateRecord`. This frozen dataclass acts as the single source of truth for candidate structures. 
**Crucial Requirement**: Always use the `parse_candidate()` factory function to instantiate a `CandidateRecord` from raw JSON/dictionary data to prevent `TypeError` from missing or unexpected fields.

---

## Evaluation Scripts

We provide a robust evaluation harness in the `eval/` directory:

1. **`validate_submission.py`**
   - Validates that the generated `submission.csv` strictly matches the hackathon constraints (100 rows, specific columns, sequential ranks, valid scores).
   - Usage: `python -m eval.validate_submission .\submission.csv`

2. **`spot_check.py`**
   - Performs manual face-validity spot checks by sampling candidates across all 10 score deciles and printing formatted profile cards.
   - Runs 5 automated checks including Disqualifier Push-Down, Reasoning Diversity, and Sentinel Signal handling.
   - Usage: `python -m eval.spot_check`

---

## Running Tests

We use `pytest` for all unit and integration tests. Ensure you run this from the project root.

```bash
pytest tests/ -v
```
*(The `pytest.ini` automatically configures `pythonpath = .` to resolve all absolute imports from `src/`.)*

---

## Streamlit Application

You can visualize the ranked candidates, their score breakdowns, and reasoning using our interactive web dashboard.

```bash
streamlit run app/streamlit_app.py
```
This application runs locally and allows you to dynamically explore the `submission.csv` output.

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
