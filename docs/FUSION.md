# Stage 5 Fusion — how the composite score actually works

This is the explainer to internalize before anyone asks you to whiteboard the
ranking formula. Everything here matches `src/ranking/fusion.py` +
`config.py` as shipped.

## The formula

```
base = 0.35 * skill_match            # cosine sim vs JD must-have vector
     + 0.25 * career_match           # cosine sim vs JD nice-to-have vector
     + 0.20 * behavioral_score       # redrob platform behavior (Member C)
     + 0.10 * availability_stability # tenure stability proxy (from career signals)
     + 0.10 * seniority_and_shipping # 0.7*seniority + 0.3*shipping

if disqualifier_sim > 0.75:          # near-zeroing penalty for red-flag profile
    base -= 0.85 * disqualifier_sim

final = clamp(base, 0, 1) * availability_multiplier   # multiplier ∈ [0.6, 1.15]
```

All five components are clamped to [0, 1] before fusion; the final score is
clamped again after the multiplier. `ScoreBreakdown` exposes every component,
so any submitted score can be decomposed line-by-line — that's why the field
is called an audit breakdown, not a black box.

## Where each number comes from

| Component | Source | Producer |
|---|---|---|
| `skill_match`, `career_match`, `disqualifier_sim` | Cosine similarity of the candidate's BGE-small embedding against three JD query vectors (must / nice / disqualifier). Retrieval backend: numpy matmul or FAISS IndexFlatIP — identical results (see `eval/retrieval_benchmark.py`) | Member B's embeddings, integrated in `rank.py` |
| `behavioral_score`, `availability_multiplier` | redrob platform signals (response rate, notice period, open-to-work) | Member C (`behavioral.py`) |
| `availability_stability` | 0.6*shipping + 0.2 baseline (+0.2 unless stale-coder), 0.5 flat for consulting-only | Derived in `fusion.py` from Member A's career signals |
| `seniority_and_shipping` | 0.7*seniority_score + 0.3*shipping_score | Derived in `fusion.py` from Member A's career signals |

## Why these weights (and how much they matter)

The weights were set by reasoning about the role (must-have skills dominate;
platform-behavior signals are supportive, not decisive), then validated — not
by learning, because the hackathon had **no labeled ground truth** (no
"good hire" labels to fit against). The honest characterization is
"informed priors, stress-tested empirically":

`eval/weight_sensitivity.py` perturbs every weight by ×0.5…×1.5
(renormalized) and re-ranks a 30k-candidate population through the real
fusion formula:

| Perturbation | Worst top-100 overlap | Worst Kendall tau (within top-100) |
|---|---|---|
| any single weight ±50% | 70% | 0.27 |
| low-weight components (stability, seniority) ±50% | 88% | 0.77 |
| plausible human tweak (skill +0.05 / behavioral −0.05) | 81% | 0.67 |

**Read:** *membership* of the top-100 is robust (≥70% overlap even under
brutal ±50% swings of the dominant weights, ≥88% for the small ones), while
*ordering within* the set is genuinely sensitive to skill/career weights.
That is the expected geometry: the top-100 boundary sits where score density
is low, but ranks inside it are close together.

## What we'd do with labels

Given even ~200 labeled "would-interview" candidates, the weights stop being
priors: fit a LightGBM ranker (already an optional dep, `RERANKER_ENABLED`)
on the same components, evaluate with NDCG@100 against a held-out split, and
keep the linear fusion as the interpretable fallback. The `ScoreBreakdown`
schema was designed so the learned model consumes the exact same features.

## Interview one-liner

> "Weights sum to 1 over five clamped components, with a near-zeroing
> disqualifier penalty and a platform-trust multiplier on top. We had no
> relevance labels, so instead of pretending to 'learn' them, we measured
> sensitivity: top-100 membership survives ±50% weight swings, and the
> breakdown dataclass keeps every score auditable. With labels, the same
> features feed a learned ranker."
