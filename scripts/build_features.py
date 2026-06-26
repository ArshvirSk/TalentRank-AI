"""
Stage 2 — Feature Engineering Entry Point.

Owner: Member A

Reads candidates.jsonl, extracts all signals, and writes the tabular
Parquet files for downstream stages.
"""

import sys
import time
import logging
from pathlib import Path
from dataclasses import asdict

import pandas as pd

# Add project root to sys.path so we can import config and src
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import config
from src.features.schema import load_candidates
from src.features.career_signals import extract_career_signals
from src.features.skill_trust import get_skill_trust_table
from src.features.honeypot import compute_honeypot_score

# ---------------------------------------------------------
# Set up logging to both console and a log file
# ---------------------------------------------------------
LOG_DIR = project_root / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "build_features.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    logging.info(f"Loading candidates from {config.CANDIDATES_PATH}...")
    
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    candidate_rows = []
    skill_rows = []

    start_time = time.time()
    count = 0

    try:
        # load_candidates is our streaming generator
        for candidate in load_candidates(config.CANDIDATES_PATH):
            count += 1
            
            # 1. Career signals
            career_signals = extract_career_signals(candidate)
            
            # 2. Honeypot check
            honeypot_score, honeypot_flag = compute_honeypot_score(candidate)
            
            # 3. Skill trust table (one-to-many)
            candidate_skills = get_skill_trust_table(candidate)
            skill_rows.extend(candidate_skills)

            # Build row for candidate_features
            row = {
                "candidate_id": candidate.candidate_id,
                **asdict(career_signals),
                "honeypot_score": honeypot_score,
                "honeypot_flag": honeypot_flag,
            }
            candidate_rows.append(row)

            if count % 10_000 == 0:
                logging.info(f"Processed {count} records...")

    except FileNotFoundError:
        logging.error(f"Could not find {config.CANDIDATES_PATH}")
        sys.exit(1)

    logging.info(f"Finished processing {count} candidates in {time.time() - start_time:.2f}s")
    
    if count == 0:
        logging.warning("No candidates processed. Exiting.")
        return

    # Write to Parquet
    logging.info(f"Writing features to {config.CANDIDATE_FEATURES_PATH}...")
    df_features = pd.DataFrame(candidate_rows)
    df_features.to_parquet(config.CANDIDATE_FEATURES_PATH, index=False)

    logging.info(f"Writing skill trust table to {config.CANDIDATE_SKILL_TRUST_PATH}...")
    df_skills = pd.DataFrame(skill_rows)
    df_skills.to_parquet(config.CANDIDATE_SKILL_TRUST_PATH, index=False)

    logging.info("Stage 2 complete.")

if __name__ == "__main__":
    main()
