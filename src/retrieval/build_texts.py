"""
person_b/build_texts.py
Converts raw candidate JSON into embeddable text strings.
Import this in both precompute_embeddings.py and rank.py.
"""

import os

import pandas as pd


def load_skill_trust(path: str) -> dict:
    """Load candidate skill trust scores from a parquet file.

    Reads the parquet file at *path* (expected to be
    ``candidate_skill_trust.parquet``) and returns a dict keyed by
    ``(candidate_id, skill_name)`` → ``trust_score`` (float).
    ``skill_name`` is normalised to lowercase before building the key.

    Returns an empty dict ``{}`` if *path* is ``None`` or the file does not
    exist — no error is raised in either case.

    Requirements: 12.1, 12.2
    """
    if path is None:
        return {}

    if not os.path.exists(path):
        return {}

    df = pd.read_parquet(path)
    return {
        (row["candidate_id"], row["skill_name"].lower()): float(row["trust_score"])
        for _, row in df.iterrows()
    }


def _assign_tier(trust_score: float) -> str:
    """Map a numeric trust score to a skill tier string.

    | Trust_Score range | Skill_Tier   |
    |-------------------|-------------|
    | ≥ 0.75            | expert       |
    | ≥ 0.50            | advanced     |
    | ≥ 0.25            | intermediate |
    | < 0.25            | beginner     |

    Requirements: 1.1, 1.2, 1.3, 1.4
    """
    if trust_score >= 0.75:
        return "expert"
    if trust_score >= 0.50:
        return "advanced"
    if trust_score >= 0.25:
        return "intermediate"
    return "beginner"


def _assign_tier_from_proficiency(proficiency: str) -> str:
    """Map a raw proficiency string to a skill tier string (case-insensitive).

    | Raw proficiency string | Skill_Tier   |
    |------------------------|-------------|
    | "Expert"               | expert       |
    | "Advanced"             | advanced     |
    | "Intermediate"         | intermediate |
    | "Beginner" / other     | beginner     |

    Requirements: 1.1, 1.2, 1.3, 1.4
    """
    normalised = proficiency.lower()
    if normalised == "expert":
        return "expert"
    if normalised == "advanced":
        return "advanced"
    if normalised == "intermediate":
        return "intermediate"
    return "beginner"


def _build_redrob_section(signals: dict) -> str | None:
    """Build the RedRob signals text block for a candidate.

    Returns None if signals is empty or absent.
    Handles sentinel values: -1 means "no data", not a low score.

    Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3, 3.4, 3.5
    """
    if not signals:
        return None

    phrases = []

    # GitHub activity score — sentinel -1 means no account
    if "github_activity_score" in signals:
        val = signals["github_activity_score"]
        if val == -1:
            phrases.append("GitHub: no account")
        elif val >= 0:
            phrases.append(f"GitHub activity score: {val}")

    # Offer acceptance rate — sentinel -1 means no offer history
    if "offer_acceptance_rate" in signals:
        val = signals["offer_acceptance_rate"]
        if val == -1:
            phrases.append("Offer history: none")
        elif val >= 0:
            phrases.append(f"Offer acceptance rate: {val}")

    # Boolean flags — only emit when True (False is silent)
    if signals.get("open_to_work_flag") is True:
        phrases.append("Actively seeking new opportunities")

    if signals.get("willing_to_relocate") is True:
        phrases.append("Willing to relocate")

    # Profile completeness — emit whenever present (not None)
    if signals.get("profile_completeness_score") is not None:
        phrases.append(f"Profile completeness: {signals['profile_completeness_score']}%")

    # Skill assessment scores — one phrase per assessment
    skill_scores = signals.get("skill_assessment_scores")
    if skill_scores:
        for key, value in skill_scores.items():
            phrases.append(f"{key} score: {value}")

    if not phrases:
        return None

    return ", ".join(phrases)


def build_candidate_text(
    c: dict,
    skill_trust_path: str | None = None,
    _trust_cache: dict | None = None,
) -> str:
    p = c["profile"]
    parts = []

    # 1. Headline + summary (most signal-dense)
    if p.get("headline"):
        parts.append(p["headline"])
    if p.get("summary"):
        parts.append(p["summary"])

    # 2. Current role context
    parts.append(
        f"Current: {p['current_title']} at {p['current_company']} "
        f"({p['current_industry']}, {p['current_company_size']} employees, "
        f"{p['years_of_experience']} years total experience)"
    )

    # 3. Career history — most important for semantic matching
    # "Built a retrieval system" lives here, not in skills
    for role in c.get("career_history", []):
        duration = f"{role['duration_months']}mo"
        current = " [current]" if role.get("is_current") else ""
        parts.append(
            f"{role['title']} at {role['company']}{current} "
            f"({duration}, {role['industry']}, {role['company_size']}): "
            f"{role['description']}"
        )

    # 4. Skills — tier-filtered blocks
    candidate_id = c["candidate_id"]

    # Build/use trust cache
    cache = _trust_cache
    if cache is None and skill_trust_path is not None:
        cache = load_skill_trust(skill_trust_path)
    if cache is None:
        cache = {}

    # Deduplicate by normalised skill name (keep first occurrence)
    seen_names = set()
    deduped_skills = []
    for s in c.get("skills", []):
        norm_name = s["name"].lower()
        if norm_name not in seen_names:
            seen_names.add(norm_name)
            deduped_skills.append(s)

    # Assign tier to each skill
    expert_adv, intermediate, beginner = [], [], []
    for s in deduped_skills:
        norm_name = s["name"].lower()
        key = (candidate_id, norm_name)
        if key in cache:
            tier = _assign_tier(cache[key])
        else:
            tier = _assign_tier_from_proficiency(s.get("proficiency", "beginner"))

        duration = s.get("duration_months", 0)
        entry = f"{s['name']} ({tier}, {duration}mo)"
        if tier in ("expert", "advanced"):
            expert_adv.append(entry)
        elif tier == "intermediate":
            intermediate.append(entry)
        else:
            beginner.append(entry)

    # Build sub-blocks
    if expert_adv:
        parts.append("Core skills: " + ", ".join(expert_adv))
    if intermediate:
        parts.append("Skills: " + ", ".join(intermediate))
    if beginner:
        parts.append("Other skills: " + ", ".join(beginner))

    # 5. RedRob signals — behavioural and platform signals
    redrob_section = _build_redrob_section(c.get("redrob_signals"))
    if redrob_section:
        parts.append(redrob_section)

    # 6. Certifications
    for cert in c.get("certifications", []):
        parts.append(f"Certified: {cert['name']} by {cert['issuer']} ({cert['year']})")

    # 7. Education
    for edu in c.get("education", []):
        parts.append(
            f"Education: {edu['degree']} in {edu['field_of_study']} "
            f"from {edu['institution']} ({edu.get('tier', 'unknown')})"
        )

    return " | ".join(filter(None, parts))


# JD decomposed into 3 semantic query texts
# These are NOT copied from the JD — they encode what the JD *means*

JD_MUST_HAVE = """
Senior AI engineer with production experience building embeddings-based retrieval systems
using sentence-transformers, BGE, E5, or OpenAI embeddings deployed to real users at product companies.
Vector database or hybrid search infrastructure experience: Pinecone, Weaviate, Qdrant, FAISS,
Elasticsearch, Milvus, or OpenSearch in production.
Strong Python. Evaluation frameworks for ranking systems: NDCG, MRR, MAP, A/B testing,
offline-to-online correlation. 5-9 years total experience, of which majority in applied ML and AI
at product companies (not consulting firms). Shipped end-to-end ranking, search, or recommendation
systems to real users at meaningful scale. Located in India or willing to relocate to Pune or Noida.
Information retrieval, semantic search, candidate matching, recommendation systems.
"""

JD_NICE_TO_HAVE = """
LLM fine-tuning experience using LoRA, QLoRA, or PEFT methods.
Learning-to-rank models such as XGBoost-based or neural LTR approaches.
Prior exposure to HR technology, recruiting platforms, or two-sided marketplace products.
Distributed systems or large-scale ML inference optimization at scale.
Open-source contributions in AI or machine learning space on GitHub.
Mentoring junior engineers. Startup or early-stage company experience.
"""

JD_DISQUALIFIER = """
Pure academic research without production deployment. No real users ever used the systems built.
Marketing manager or sales role or non-engineering title despite AI keywords listed.
Only experience with LangChain tutorials and OpenAI API calls in last 12 months with no prior ML history.
Entire career spent only at TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini with no product company experience.
Primary expertise in computer vision, image classification, speech recognition, or robotics without NLP or IR.
Senior architect or tech lead who has not written production code in 18 months.
Frequent company switches every 1-2 years purely for title escalation.
Customer support, content writer, business analyst, or operations role.
"""
