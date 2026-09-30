# File-Level Ownership (Team Coder's Block)

Authoritative map of who wrote what, from git history (`git log --format='%an' -- <path>`).
Use this when asked "which parts did you personally write?"

## Arshvir Singh Kalsi (Member A — Data Engineering & Feature Pipeline; also team integrator)

| Path | What it is |
|---|---|
| `config.py` | Global configuration: paths, scoring weights, constraints, keyword maps |
| `src/features/schema.py` | Typed dataclass schema + streaming JSONL loader for 100k records |
| `src/features/career_signals.py` | Career-signal extraction (employer typing, production evidence, flag heuristics) |
| `src/features/skill_trust.py` | Skill trust scoring (proficiency × duration × endorsements, padding discount) |
| `src/features/honeypot.py` | Rule-based fraud/sentinel-profile detection |
| `scripts/build_features.py` | Feature precompute pipeline (JSONL → parquet) |
| `src/ranking/rank.py` | Final ranking CLI — integrates stages 2–8 into the submission writer |
| `src/retrieval/faiss_index.py` | FAISS IndexFlatIP build/query + numpy backend (post-hackathon implementation of the original scaffold) |
| `src/retrieval/jd_parser.py` | Working .docx JD parser (post-hackathon implementation of the original scaffold) |
| `tests/` (majority) | Test suite across all members' modules |
| `eval/weight_sensitivity.py` | Weight-perturbation stability analysis (post-hackathon) |
| `eval/retrieval_benchmark.py` | numpy-vs-FAISS measured comparison (post-hackathon) |

## Siddhant Sawant (Member B — Semantic Search & Retrieval)

| Path | What it is |
|---|---|
| `personB/` (his working copies) | `precompute_embeddings.py`, `build_texts.py`, `compute_semantic_scores.py`, `benchmark_models.py` |
| `src/retrieval/embed.py` | Parallel checkpointed embedding precompute (BGE-small, 100k × 384) |
| `src/retrieval/build_texts.py` | Candidate→text rendering + the hand-decomposed JD query texts |
| JD decomposition | `JD_MUST_HAVE` / `JD_NICE_TO_HAVE` / `JD_DISQUALIFIER` semantics |

## Viraj Prabhu (Member C — Ranking Engine & ML Scoring)

| Path | What it is |
|---|---|
| `src/ranking/fusion.py` | Composite scoring formula (weights, disqualifier penalty, availability multiplier) |
| `src/ranking/behavioral.py` | Behavioral score + availability multiplier from redrob_signals |
| `tests/test_member_c.py` | Tests for the fusion/behavioral modules |

## Ghrani Ganesh Poojari (Member D — Frontend, Eval, Docs & Submission)

| Path | What it is |
|---|---|
| `app/streamlit_app.py` | Interactive results dashboard |
| `src/explain/templates.py` | Reasoning-template banks |
| `eval/spot_check.py` | Face-validity spot checks |
| `submission_metadata.yaml` | Competition metadata |

## Integration notes

- `src/ranking/rank.py` is Member C's scoring logic orchestrated by Member A's
  integration: loaders, similarity computation, honeypot gating, and CSV writing
  were assembled by Arshvir from all members' artifacts.
- `src/explain/generate.py` is Member D's module, integrated unchanged; the
  JD-aware segment inside it activates only when a real `JDProfile` is supplied
  (which `rank.py` now provides via `jd_parser.py`).
- The original `src/retrieval/faiss_index.py` and `jd_parser.py` were scaffolds
  (`NotImplementedError`) submitted under time constraints; the working
  implementations landed post-hackathon and are labeled as such in their
  module docstrings.
