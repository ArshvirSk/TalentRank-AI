# TalentRank AI

> Submission for the **India Runs Data & AI Challenge: Intelligent Candidate Discovery & Ranking**
> By **Team ASK**

This repository contains an intelligent ranking engine capable of evaluating 100,000 candidate profiles against a Job Description (JD) for an ML Engineer role, adhering to strict offline, CPU-only, and 5-minute execution constraints.

---

## 🎯 The Challenge (What Was to Be Done)

The core requirements and constraints were extremely strict:
1. **Scale:** Process and rank **100,000 candidate profiles** (provided as JSONL).
2. **Hardware:** Must run entirely **offline** on a **CPU-only** machine. No cloud APIs (like OpenAI) or GPU inference allowed at runtime.
3. **Performance:** The entire ranking pipeline had to complete in **under 5 minutes**.
4. **Memory:** Must consume **≤ 16 GB RAM**.
5. **Output:** Generate a `submission.csv` containing the top 100 candidates, complete with their scores and a human-readable "reasoning" string explaining *why* they were selected.

---

## 🚀 Results Achieved

We successfully built a robust, production-ready system that perfectly adhered to all hackathon constraints.

1. **Lightning Fast Performance:** The final pipeline executes in roughly **13 seconds** (well under the 5-minute limit), processing all 100,000 candidates efficiently using CPU-only vector math.
2. **High-Quality Output:** The final `submission.csv` contains a perfectly curated list of the top 100 candidates. 
3. **Score Distribution:** We implemented dynamic min-max normalization for behavioral scores and optimal weighting to ensure a healthy, differentiated score distribution ranging from **0.73 to 0.86** for the top candidates.
4. **Automated Verification Passed:** Our automated checks guarantee 0 disqualifying profiles in the top ranks, high title diversity, and 100% unique reasoning strings.

---

## 🛠️ The Architecture & Pipeline (How We Did It)

To meet the strict CPU and time constraints, we designed a pipeline heavily reliant on **precomputation and vector similarity**.

```text
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

### 1. Data Engineering & Precomputation
Instead of running heavy NLP models on 100,000 candidates at runtime, we pre-processed the data.
* **Embeddings:** We used `SentenceTransformers` to generate 384-dimensional vector embeddings for candidate skills, job titles, and the JD components. These were saved as `.npy` arrays.
* **Feature Extraction:** We parsed the unstructured candidate JSONs to extract numerical features (months of experience per skill, notice period length, response rate) and saved them as highly optimized `.parquet` files.

### 2. Fast Retrieval (The Funnel)
* We implemented **FAISS (Facebook AI Similarity Search)** to quickly compute cosine similarities between the precomputed JD vectors and the candidate vectors.
* Instead of scoring all 100,000 candidates thoroughly, we used FAISS to instantly shortlist the top 500 candidates based on their "must-have" skill alignment.

### 3. Multi-Factor Ranking Engine
The shortlisted candidates were passed through a complex scoring engine that evaluated them across five normalized dimensions:
* **Skill Score (Weight: 45%):** Similarity to the JD's technical requirements.
* **Career Score (Weight: 20%):** Alignment of their job title and total years of experience.
* **Behavioral Score (Weight: 15%):** A min-max normalized score capturing their responsiveness, notice period, and "open to work" status.
* **Stability (Weight: 10%):** Evaluated their job tenure to ensure they aren't chronic job-hoppers.
* **Seniority (Weight: 10%):** Rewarded candidates closer to the sweet spot of 4-6 years of experience.
* **Disqualifiers:** We applied severe penalties to candidates missing critical requirements or possessing red flags (e.g., consulting-only profiles).

### 4. Fraud Detection
We built a `honeypot.py` module to dynamically filter out fake, sentinel, or corrupted profiles from the dataset before final ranking.

---

## 💻 Quick Start

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

## 👥 Team — ASK

| Member | Name                    | Responsibility                         | Directory        | Stages |
|--------|-------------------------|----------------------------------------|------------------|--------|
| A      | **Arshvir Singh Kalsi** | Data Engineering & Feature Pipeline    | `src/features/`  | 2, 6   |
| B      | **Siddhant Sawant**     | Semantic Search & Retrieval            | `src/retrieval/`  | 1, 3, 4|
| C      | **Viraj Prabhu**        | Ranking Engine & ML Scoring            | `src/ranking/`    | 5, 7   |
| D      | **Ghrani Ganesh Poojari** | Frontend, Eval, Docs & Submission    | `app/`, `eval/`   | 8      |

> See [CONTRIBUTORS.md](CONTRIBUTORS.md) for detailed per-member contribution breakdown.

---

## 📊 Data Schema Validation

The system strictly enforces data correctness using `src.features.schema.CandidateRecord`. This frozen dataclass acts as the single source of truth for candidate structures. 
**Crucial Requirement**: Always use the `parse_candidate()` factory function to instantiate a `CandidateRecord` from raw JSON/dictionary data to prevent `TypeError` from missing or unexpected fields.

---

## 🧪 Evaluation & Testing

We provide a robust evaluation harness in the `eval/` directory and use `pytest` for all unit testing.

```bash
# Run unit tests
pytest tests/ -v

# Validate the generated submission.csv strictly matches constraints
python -m eval.validate_submission .\submission.csv

# Perform manual face-validity spot checks
python -m eval.spot_check
```

---

## 📂 Project Structure

```
TalentRankAI/
├── README.md
├── requirements.txt
├── Dockerfile
├── .gitignore
├── CONTRIBUTORS.md
├── submission_metadata.yaml
├── config.py
├── data/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── features/
│   │   ├── schema.py
│   │   ├── career_signals.py
│   │   ├── skill_trust.py
│   │   └── honeypot.py
│   ├── retrieval/
│   │   ├── jd_parser.py
│   │   ├── embed.py
│   │   └── faiss_index.py
│   ├── ranking/
│   │   ├── fusion.py
│   │   ├── behavioral.py
│   │   ├── reranker.py
│   │   └── rank.py
│   └── explain/
│       ├── templates.py
│       └── generate.py
├── app/
│   └── streamlit_app.py
├── eval/
│   ├── spot_check.py
│   └── validate_submission.py
└── tests/
    └── ...
```
