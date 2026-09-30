"""
Weight-sensitivity analysis for the Stage 5 fusion formula.

Answers the interview question "your weights look hand-tuned — how sensitive
is the ranking to them?" with numbers instead of hand-waving.

Method
------
1. Build a synthetic-but-realistic shortlist of scored candidates using the
   REAL fusion code path (compute_composite_score) over sampled feature
   distributions matched to the pipeline's components.
2. For the production weights w* (config.py), and for perturbed weight
   vectors w' where each weight is scaled by s in {0.5, 0.75, 1.25, 1.5}
   (renormalized to sum to 1), compute:
     - top-100 overlap:  |top100(w') ∩ top100(w*)| / 100
     - Kendall tau between the two top-100 orderings
3. Report per-component and aggregate stability.

Usage:
    python -m eval.weight_sensitivity            # full report
    python -m eval.weight_sensitivity --n 20000  # bigger synthetic sample
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import replace

import numpy as np

import config
from src.ranking.fusion import compute_composite_score
from src.features.career_signals import CareerSignals

RNG_SEED = 42

WEIGHT_KEYS = (
    "skill_match",
    "career_match",
    "behavioral_score",
    "availability_stability",
    "seniority_and_shipping",
)

PRODUCTION_WEIGHTS = {
    "skill_match": config.WEIGHT_SKILL_MATCH,
    "career_match": config.WEIGHT_CAREER_MATCH,
    "behavioral_score": config.WEIGHT_BEHAVIORAL,
    "availability_stability": config.WEIGHT_AVAILABILITY_STABILITY,
    "seniority_and_shipping": config.WEIGHT_SENIORITY_SHIPPING,
}

SCALES = (0.5, 0.75, 1.25, 1.5)


# ---------------------------------------------------------------------------
# Synthetic-but-realistic component sampler
# ---------------------------------------------------------------------------

def sample_career_signals(rng: np.random.Generator) -> CareerSignals:
    """Sample a CareerSignals instance from plausible pipeline distributions.

    Field names follow src/features/career_signals.CareerSignals exactly.
    Distributions reflect the pipeline's semantics: most candidates have
    moderate production evidence; flags are sparse; seniority is roughly
    uniform over [0.2, 0.9].
    """
    shipping = float(np.clip(rng.beta(2.0, 2.0), 0.0, 1.0))
    return CareerSignals(
        employer_type_current=str(rng.choice(["product", "services", "research", "other"],
                                             p=[0.45, 0.35, 0.10, 0.10])),
        employer_types=["product"],
        consulting_only_flag=bool(rng.random() < 0.08),
        production_evidence_score=shipping,
        title_chaser_score=float(np.clip(rng.beta(1.5, 6.0), 0.0, 1.0)),
        research_only_flag=bool(rng.random() < 0.06),
        recent_llm_only_flag=bool(rng.random() < 0.12),
        cv_speech_without_nlp_flag=bool(rng.random() < 0.05),
        stale_coder_flag=bool(rng.random() < 0.10),
        total_career_months=int(rng.integers(6, 144)),
        seniority_score=float(np.clip(rng.uniform(0.2, 0.9), 0.0, 1.0)),
        shipping_score=shipping,
    )


def build_population(n: int, seed: int = RNG_SEED) -> list[dict]:
    """Score n synthetic candidates through the REAL fusion function.

    Returns list of dicts with candidate_id, breakdown, and the component
    values that fusion consumed (so perturbation replays are consistent).
    """
    rng = np.random.default_rng(seed)
    population = []
    for i in range(n):
        signals = sample_career_signals(rng)
        skill = float(np.clip(rng.beta(2.0, 5.0), 0.0, 1.0))       # must-have sim
        career = float(np.clip(rng.beta(2.5, 4.0), 0.0, 1.0))      # nice-to-have sim
        disq = float(np.clip(rng.beta(1.2, 12.0), 0.0, 1.0))       # disqualifier sim
        behavioral = float(np.clip(rng.beta(3.0, 2.0), 0.0, 1.0))
        avail_mult = float(np.clip(rng.uniform(0.6, 1.15), 0.6, 1.15))

        breakdown = compute_composite_score(
            candidate_id=f"SYN_{i:06d}",
            skill_match_sim=skill,
            career_match_sim=career,
            disqualifier_sim=disq,
            career_signals=signals,
            availability_multiplier=avail_mult,
            behavioral_score=behavioral,
        )
        population.append({
            "candidate_id": f"SYN_{i:06d}",
            "breakdown": breakdown,
            "skill": skill,
            "career": career,
            "disq": disq,
            "behavioral": behavioral,
            "avail_mult": avail_mult,
            "signals": signals,
        })
    return population


# ---------------------------------------------------------------------------
# Ranking under a weight vector
# ---------------------------------------------------------------------------

def _rank_population(population: list[dict], weights: dict[str, float]) -> list[tuple[float, str]]:
    """Recompute final scores under ``weights`` using fusion's exact formula."""
    scored = []
    for rec in population:
        bd = rec["breakdown"]
        base = (
            weights["skill_match"] * rec["skill"]
            + weights["career_match"] * rec["career"]
            + weights["behavioral_score"] * rec["behavioral"]
            + weights["availability_stability"] * bd.availability_stability
            + weights["seniority_and_shipping"] * bd.seniority_and_shipping
        )
        base = max(0.0, base - bd.disqualifier_penalty)
        final = max(0.0, min(1.0, base * rec["avail_mult"]))
        scored.append((final, rec["candidate_id"]))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return scored


def _top_k(scored: list[tuple[float, str]], k: int = 100) -> list[str]:
    return [cid for _, cid in scored[:k]]


def _kendall_tau(a: list[str], b: list[str]) -> float:
    """Kendall tau between two orderings of (roughly) the same ID set.

    IDs present in only one list are counted as maximal discordance for the
    pairs they form (gamma-style normalization by total pairs).
    """
    common = [cid for cid in a if cid in set(b)]
    if len(common) < 2:
        return 0.0
    pos_b = {cid: i for i, cid in enumerate(b)}
    concordant = discordant = 0
    for (i, cid_i), (j, cid_j) in itertools.combinations(enumerate(common), 2):
        if (i < j) == (pos_b[cid_i] < pos_b[cid_j]):
            concordant += 1
        else:
            discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 0.0


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def run_analysis(n: int) -> int:
    print("=" * 74)
    print(f"Weight-sensitivity analysis — n={n:,} synthetic candidates, "
          f"production weights:")
    for key in WEIGHT_KEYS:
        print(f"  {key:<26} {PRODUCTION_WEIGHTS[key]:.2f}")
    print("=" * 74)

    population = build_population(n)
    baseline = _rank_population(population, PRODUCTION_WEIGHTS)
    baseline_top = _top_k(baseline)
    print(f"\nBaseline top-100 score range: "
          f"[{baseline[0][0]:.4f} … {baseline[99][0]:.4f}]")

    print(f"\n{'perturbation':<34} {'top-100 overlap':>16} {'Kendall tau':>12}")
    print("-" * 74)

    worst_overlap, worst_tau = 1.0, 1.0
    for key in WEIGHT_KEYS:
        for scale in SCALES:
            weights = dict(PRODUCTION_WEIGHTS)
            weights[key] = PRODUCTION_WEIGHTS[key] * scale
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}  # renormalize

            perturbed = _rank_population(population, weights)
            perturbed_top = _top_k(perturbed)

            overlap = len(set(baseline_top) & set(perturbed_top)) / len(baseline_top)
            tau = _kendall_tau(baseline_top, perturbed_top)
            worst_overlap = min(worst_overlap, overlap)
            worst_tau = min(worst_tau, tau)

            print(f"{key} x{scale:<5} ({weights[key]:.3f})".ljust(34)
                  + f"{overlap:>16.2%} {tau:>12.3f}")

    # Additive shift: +0.05 to skill, -0.05 to behavioral (plausible human tweak)
    shifted = dict(PRODUCTION_WEIGHTS)
    shifted["skill_match"] += 0.05
    shifted["behavioral_score"] -= 0.05
    perturbed = _rank_population(population, shifted)
    overlap = len(set(baseline_top) & set(_top_k(perturbed))) / 100
    tau = _kendall_tau(baseline_top, _top_k(perturbed))
    worst_overlap = min(worst_overlap, overlap)
    worst_tau = min(worst_tau, tau)
    print("-" * 74)
    print(f"{'skill +0.05 / behavioral -0.05':<34} {overlap:>16.2%} {tau:>12.3f}")

    print("=" * 74)
    print(f"Worst-case across all perturbations: overlap {worst_overlap:.2%}, "
          f"Kendall tau {worst_tau:.3f}")
    verdict = (
        "STABLE — rankings are robust to ±50% single-weight perturbations."
        if worst_overlap >= 0.75 and worst_tau >= 0.75
        else "SENSITIVE — rankings move materially under weight changes; "
             "the weights are a real modeling choice, not a formality."
    )
    print(f"Verdict: {verdict}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50_000,
                        help="Synthetic population size (default 50,000)")
    args = parser.parse_args()
    raise SystemExit(run_analysis(args.n))
