# Person B — Semantic Layer & Embeddings

## Table of Contents
1. [What This System Is](#1-what-this-system-is)
2. [How the System Works — The Core Idea](#2-how-the-system-works--the-core-idea)
3. [Why Three Query Vectors?](#3-why-three-query-vectors)
4. [Project Structure](#4-project-structure)
5. [Data Explained in Detail](#5-data-explained-in-detail)
6. [Each File Explained](#6-each-file-explained)
7. [How to Install Dependencies](#7-how-to-install-dependencies)
8. [How to Execute — Step by Step](#8-how-to-execute--step-by-step)
9. [CLI Reference](#9-cli-reference)
10. [Input and Output Behaviour](#10-input-and-output-behaviour)
11. [What Good Output Looks Like](#11-what-good-output-looks-like)
12. [How This Fits Into the Larger System](#12-how-this-fits-into-the-larger-system)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. What This System Is

The overall system ranks **100,000 candidate profiles** against a single job description (Senior AI/ML Engineer, Pune or Noida, India) and produces a shortlist of the top 100 candidates, each with a score and reasoning.

**Person B's specific responsibility:** Convert every candidate profile and the job description into numbers (called embeddings) that capture *meaning*, then measure how semantically similar each candidate is to the job requirements.

The core problem this solves:
- Traditional keyword search misses candidates who describe the right experience using different words (e.g., "built a semantic search layer" instead of "RAG pipeline")
- Traditional keyword search promotes candidates who stuffed their resume with trending keywords without ever doing the work
- This system understands meaning, not just character matches

---

## 2. How the System Works — The Core Idea

### The Embedding Concept

An **embedding** is a list of 384 numbers that represents the meaning of a piece of text. Two texts that mean the same thing — even if they use completely different words — produce similar number lists. Two texts that mean opposite things produce very different number lists.

Example:
- "built a vector search system using FAISS" → [0.12, -0.34, 0.87, ...]
- "developed a semantic retrieval pipeline with dense embeddings" → [0.11, -0.33, 0.86, ...] ← very close!
- "managed sales team and hit quarterly targets" → [-0.45, 0.23, -0.12, ...] ← very different

**Cosine similarity** (the dot product of two L2-normalized vectors) measures how close two embeddings are. A score of 1.0 means identical meaning. A score of 0.0 means completely unrelated.

### The Two-Phase Pipeline

```
PHASE 1 — Precompute (run ONCE, ~37 min for 100k)
  candidates.jsonl
       ↓
  build_candidate_text()    ← converts JSON → rich text string
       ↓
  BGE-small model           ← converts text → 384 numbers
       ↓
  candidate_embeddings.npy  ← (100000, 384) matrix saved to disk

PHASE 2 — Scoring (run in seconds)
  candidate_embeddings.npy  ← load saved matrix
  jd_query_vectors.npy      ← load 3 pre-embedded JD queries
       ↓
  matrix multiply (dot product)
       ↓
  semantic_scores.parquet   ← 5 columns per candidate
```

Phase 1 is the slow step. It only runs once. Phase 2 is milliseconds — pure NumPy math.

---

## 3. Why Three Query Vectors?

A single embedding of the full job description mixes everything together — the must-haves, nice-to-haves, and disqualifiers all blend into one vector and cancel each other out.

Instead, the JD is decomposed into **three separate paragraphs**, each embedded independently:

| Query | Purpose | Effect on ranking |
|-------|---------|-------------------|
| `JD_MUST_HAVE` | Core requirements: production embeddings, vector DBs, 5-9 yrs exp, India | Higher = stronger candidate |
| `JD_NICE_TO_HAVE` | Bonus signals: LLM fine-tuning, LTR, open-source, startup exp | Higher = bonus points |
| `JD_DISQUALIFIER` | Red flags: academic-only, non-engineering roles, pure consulting, CV keywords | Higher = penalise this candidate |

### Why this catches the two hardest cases

**False negative (legitimate engineer, missed by keyword search):**
A real ML engineer who writes "built semantic search over 10M documents" but never uses the word "RAG" or "vector database" will score high on `sim_must` (meaning matches) and low on `sim_disq` → correctly ranked highly.

**False positive (keyword stuffer, promoted by keyword search):**
A Marketing Manager who listed "Pinecone, LangChain, GPT-4" in their skills will score high on `sim_must` (keywords match) BUT also high on `sim_disq` (their actual career profile matches the disqualifier description) → net score drops → correctly ranked lower.

### The Three JD Query Texts (what they contain)

**MUST_HAVE text** — describes: production embeddings (sentence-transformers, BGE, E5, OpenAI), vector databases (Pinecone, Weaviate, Qdrant, FAISS, Milvus, Elasticsearch), strong Python, evaluation frameworks (NDCG, MRR, MAP, A/B testing), 5-9 years at product companies, shipped ranking/search/recommendation systems, India location.

**NICE_TO_HAVE text** — describes: LLM fine-tuning (LoRA, QLoRA, PEFT), learning-to-rank (XGBoost LTR, neural LTR), HR tech exposure, distributed ML inference, GitHub open-source contributions, mentoring, startup experience.

**DISQUALIFIER text** — describes: pure academic research, marketing/sales/non-engineering roles, LangChain-only experience with no prior ML history, entire career at TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini, computer vision/speech without NLP, senior architects who stopped coding, frequent job-hopping for title escalation.

---

## 4. Project Structure

```
Hackathon/
├── data/                              ← Input data (read-only)
│   ├── candidates.jsonl               ← Full dataset: 100,000 candidate profiles (465 MB)
│   ├── sample_candidates.json         ← 50-candidate sample for testing
│   ├── candidate_schema.json          ← JSON schema describing each candidate record
│   ├── candidate_skill_trust.parquet  ← Pre-computed skill trust scores (in data/processed/)
│   ├── job_description.docx           ← The actual job description document
│   ├── redrob_signals_doc.docx        ← Documentation for the 23 redrob_signals fields
│   ├── submission_spec.docx           ← Rules for final hackathon submission format
│   ├── submission_metadata_template.yaml
│   ├── sample_submission.csv          ← Example of what the final output should look like
│   ├── validate_submission.py         ← Script to validate final submission format
│   └── README.docx
│   └── processed/
│       ├── candidate_features.parquet ← Person A's structured feature extractions
│       └── candidate_skill_trust.parquet ← Pre-built skill trust weights
│
└── personB/                           ← Person B's code (this folder)
    ├── build_texts.py                 ← Text construction: JSON → embeddable string
    ├── precompute_embeddings.py       ← Phase 1: embed all candidates, save matrix
    ├── compute_semantic_scores.py     ← Phase 2: dot product scoring, save parquet
    ├── benchmark_models.py            ← Utility: compare embedding model speeds
    ├── test_build_texts.py            ← Test suite: 21 tests covering all logic
    ├── conftest.py                    ← Pytest configuration and marks
    ├── README.md                      ← This file
    ├── precomputed/                   ← Generated by precompute_embeddings.py
    │   ├── candidate_embeddings.npy   ← (100000, 384) float32 matrix (153 MB)
    │   ├── candidate_index.parquet    ← Row number → candidate_id mapping
    │   └── jd_query_vectors.npy       ← (3, 384) float32 — [must, nice, disq]
    ├── precomputed_sample/            ← Same but for 50-candidate sample run
    │   ├── candidate_embeddings.npy
    │   ├── candidate_index.parquet
    │   └── jd_query_vectors.npy
    ├── semantic_scores.parquet        ← FINAL OUTPUT — hand to Person C (full 100k)
    └── semantic_scores_sample.parquet ← Sample output for testing
```

---

## 5. Data Explained in Detail

### 5.1 candidates.jsonl — The Main Dataset

**Format:** JSONL (JSON Lines) — one complete JSON object per line. 100,000 lines = 100,000 candidates. File size: ~465 MB.

Each candidate record has exactly five top-level keys:

#### Key 1: `profile` — Identity and career summary
```json
{
  "candidate_id": "CAND_0000001",
  "profile": {
    "anonymized_name": "Ira Vora",
    "headline": "Backend Engineer | SQL, Spark, Cloud",
    "summary": "Software professional with 6.9 years...",
    "location": "Toronto",
    "country": "Canada",
    "years_of_experience": 6.9,
    "current_title": "Backend Engineer",
    "current_company": "Mindtree",
    "current_company_size": "10001+",
    "current_industry": "IT Services"
  }
}
```
- `candidate_id`: Unique identifier string, format `CAND_XXXXXXX`
- `headline`: One-line professional summary shown at the top of their profile — high signal density
- `summary`: Multi-sentence self-description — often reveals career direction and intent
- `years_of_experience`: Float, total years (not just at current company)
- `current_company_size`: One of: `"1-10"`, `"11-50"`, `"51-200"`, `"201-500"`, `"501-1000"`, `"1001-5000"`, `"5001-10000"`, `"10001+"`
- `current_industry`: Free text, e.g., `"IT Services"`, `"Software"`, `"Manufacturing"`

#### Key 2: `career_history` — Work experience (most important for embeddings)
```json
{
  "career_history": [
    {
      "company": "Mindtree",
      "title": "Backend Engineer",
      "start_date": "2024-03-08",
      "end_date": null,
      "duration_months": 27,
      "is_current": true,
      "industry": "IT Services",
      "company_size": "10001+",
      "description": "Implemented streaming data pipelines on Kafka..."
    }
  ]
}
```
- `description`: **The most important field in the entire dataset for embedding quality.** This is a paragraph where a candidate describes what they actually *did* in the role. A candidate who built a vector search system describes it here. They may not use the word "RAG" but they will say "built a semantic search layer using FAISS and sentence-transformers" — the embedding model is designed to catch this.
- `duration_months`: Integer — how long they stayed at this job
- `is_current`: Boolean — whether this is their active role (`end_date` is null for current roles)
- `start_date` / `end_date`: ISO date strings, `null` if still in role

#### Key 3: `skills` — Skills with trust signals
```json
{
  "skills": [
    {
      "name": "Milvus",
      "proficiency": "advanced",
      "endorsements": 40,
      "duration_months": 35
    }
  ]
}
```
- `proficiency`: One of `"beginner"`, `"intermediate"`, `"advanced"`, `"expert"`
- `endorsements`: Integer — how many people endorsed this skill on the platform
- `duration_months`: How long they have been using this skill

**Important:** Skills alone are unreliable. Anyone can claim "Pinecone" as a skill. The system weights skills by combining proficiency + endorsements + duration into a trust score (from `candidate_skill_trust.parquet`).

#### Key 4: `education` — Degrees with institution tier
```json
{
  "education": [
    {
      "institution": "Lovely Professional University",
      "degree": "B.E.",
      "field_of_study": "Computer Science",
      "start_year": 2017,
      "end_year": 2020,
      "grade": "8.24 CGPA",
      "tier": "tier_3"
    }
  ]
}
```
- `tier`: Pre-computed institution quality tier. Values: `"tier_1"` (IITs, IISc, top global), `"tier_2"` (NITs, good state universities), `"tier_3"` (other recognized universities), `"tier_4"` (unknown or unranked)
- `degree`: Abbreviated, e.g., `"B.E."`, `"M.Tech"`, `"Ph.D"`, `"B.Sc"`, `"M.Sc"`
