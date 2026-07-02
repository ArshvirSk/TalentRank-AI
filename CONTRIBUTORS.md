# Contributors

## Team: Coder's Block

This project was built collaboratively by a 4-member team for the
**India Runs Data & AI Challenge — Intelligent Candidate Discovery & Ranking**.

| Member | Name                    | Role                                | Stages  | Key Files                                          |
|--------|-------------------------|-------------------------------------|---------|-----------------------------------------------------|
| A      | Arshvir Singh Kalsi     | Data Engineering & Feature Pipeline | 2, 6    | `src/features/`, `scripts/build_features.py`        |
| B      | Siddhant Sawant         | Semantic Search & Retrieval         | 1, 3, 4 | `src/retrieval/`, `data/`                           |
| C      | Viraj Prabhu            | Ranking Engine & ML Scoring         | 5, 7    | `src/ranking/`, `src/explain/`                      |
| D      | Ghrani Ganesh Poojari   | Frontend, Eval, Docs & Submission   | 8       | `app/`, `eval/`, `tests/`                           |

## Contribution Summary

### Member A — Arshvir Singh Kalsi
- Designed the 8-stage pipeline architecture and project scaffold
- Built the `CandidateRecord` schema and `parse_candidate()` factory
- Implemented career signal extraction (consulting-only, title-chaser, stale-coder, research-only flags)
- Built the honeypot/fraud detection module
- Integrated all member branches into `main` and resolved cross-module dependencies
- Led testing and final submission

### Member B — Siddhant Sawant
- Built the JD parser (`jd_parser.py`) to extract must-have, nice-to-have, and disqualifier requirements
- Implemented candidate text construction (`build_texts.py`) for embedding input
- Set up the sentence-transformers embedding pipeline (`embed.py`) using BGE-small-en-v1.5
- Precomputed 100k candidate embeddings and JD query vectors (`.npy` artifacts)
- Built the FAISS index for fast approximate nearest-neighbor retrieval

### Member C — Viraj Prabhu
- Designed the composite scoring formula (fusion.py) with weighted skill, career, behavioral, and seniority components
- Implemented the behavioral scoring module using all 23 Redrob platform signals
- Built the availability multiplier (notice period, open-to-work, recruiter response rate)
- Implemented the final ranking CLI (`rank.py`) with 5-minute time constraint enforcement
- Created the explanation generator for per-candidate reasoning strings

### Member D — Ghrani Ganesh Poojari
- Built the `validate_submission.py` evaluation script (CSV format compliance)
- Created the `spot_check.py` face-validity harness (decile sampling + 5 automated checks)
- Developed the Streamlit dashboard (`app/streamlit_app.py`) for interactive exploration
- Wrote unit and integration tests across all modules
