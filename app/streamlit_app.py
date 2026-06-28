"""
TalentRank AI -- Streamlit Sandbox Demo.

Owner: Member 4

A small-sample interactive demo per submission spec section 10.5.
Allows judges to explore a subset of ranked candidates, view score
breakdowns, and understand the pipeline's reasoning.

Usage:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on sys.path so config.py is importable
# when Streamlit runs the file from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

import config


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_submission(path: str) -> pd.DataFrame | None:
    """Load submission CSV from a file path. Returns None on any error."""
    try:
        df = pd.read_csv(path)
        df["rank"] = df["rank"].astype(int)
        df["score"] = df["score"].astype(float)
        return df.sort_values("rank").reset_index(drop=True)
    except Exception:
        return None


@st.cache_resource
def load_candidates(path: str) -> dict:
    """
    Load candidates.jsonl into a dict keyed by candidate_id.
    Cached after first load -- 100k records take ~5s, subsequent loads
    are instant.
    """
    lookup: dict = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    cid = record.get("candidate_id")
                    if cid:
                        lookup[cid] = record
    except Exception:
        pass
    return lookup


def load_submission_from_upload(file) -> pd.DataFrame | None:
    """Load submission CSV from a Streamlit uploaded file object."""
    try:
        df = pd.read_csv(file)
        df["rank"] = df["rank"].astype(int)
        df["score"] = df["score"].astype(float)
        return df.sort_values("rank").reset_index(drop=True)
    except Exception:
        return None


def fmt_sentinel(val, decimals: int = 2) -> str:
    """Return 'N/A' for -1 sentinel values, else formatted string."""
    if val == -1 or val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    """Launch the Streamlit sandbox demo."""
    st.set_page_config(
        page_title="TalentRank AI -- Candidate Ranking Demo",
        page_icon="🏆",
        layout="wide",
    )

    st.title("TalentRank AI")
    st.subheader("Intelligent Candidate Discovery & Ranking")

    st.markdown(
        """
        This demo showcases the ranking pipeline's output on a small sample.
        Upload a submission CSV or load the default to explore ranked candidates.
        """
    )

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    st.sidebar.header("Configuration")

    csv_file = st.sidebar.file_uploader(
        "Upload submission.csv", type=["csv"]
    )
    candidates_file = st.sidebar.file_uploader(
        "Upload candidates.jsonl (optional -- for profile details)",
        type=["jsonl", "json"],
    )
    top_n = st.sidebar.slider(
        "Candidates to display",
        min_value=10,
        max_value=100,
        value=20,
        step=10,
    )
    show_details = st.sidebar.checkbox(
        "Show per-candidate detail panels", value=False
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Pipeline Info**")
    st.sidebar.markdown(f"Model: `{config.EMBEDDING_MODEL_NAME}`")
    st.sidebar.markdown(f"Top-K: `{config.TOP_K}`")
    st.sidebar.markdown(f"Max candidates: `{config.TOTAL_CANDIDATES:,}`")

    # ------------------------------------------------------------------
    # SECTION 0 -- Load data
    # ------------------------------------------------------------------
    if csv_file is not None:
        df = load_submission_from_upload(csv_file)
    else:
        df = load_submission(str(config.SUBMISSION_PATH))

    if candidates_file is not None:
        candidates_lookup: dict = {}
        try:
            for line in candidates_file.read().decode("utf-8").splitlines():
                line = line.strip()
                if line:
                    record = json.loads(line)
                    cid = record.get("candidate_id")
                    if cid:
                        candidates_lookup[cid] = record
        except Exception as exc:
            st.warning(f"Could not parse uploaded candidates file: {exc}")
            candidates_lookup = {}
    else:
        candidates_lookup = load_candidates(str(config.CANDIDATES_PATH))

    if df is None:
        st.warning(
            "No submission.csv found. Run the full pipeline first:\n\n"
            "```\n"
            "python -m src.ranking.rank "
            "--candidates ./data/candidates.jsonl "
            "--jd ./data/job_description.docx "
            "--out ./submission.csv\n"
            "```"
        )
        st.stop()

    # ------------------------------------------------------------------
    # SECTION 1 -- Metrics row
    # ------------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Ranked", len(df))
    col2.metric("Top Score", f"{df['score'].max():.4f}")
    col3.metric("Median Score", f"{df['score'].median():.4f}")
    col4.metric("Score Std Dev", f"{df['score'].std():.4f}")

    # ------------------------------------------------------------------
    # SECTION 2 -- Score distribution
    # ------------------------------------------------------------------
    # st.markdown("### Score Distribution")
    # score_hist = pd.cut(df["score"], bins=10).value_counts().sort_index()
    # score_hist_df = pd.DataFrame(
    #     {
    #         "score_range": [str(i) for i in score_hist.index],
    #         "count": score_hist.values,
    #     }
    # ).set_index("score_range")
    # st.bar_chart(score_hist_df)
    # st.caption(
    #     "Distribution of composite scores across all ranked candidates"
    # )
    # Use rank as index for the line chart
    st.markdown("### Score Trend by Rank")
    score_trend_df = (
        df[["rank", "score"]]
        .sort_values("rank")
        .set_index("rank")
    )

    st.line_chart(score_trend_df)

    st.caption(
        "How candidate scores change across the ranking. "
        "A steeper decline indicates stronger separation between top and lower-ranked candidates."
    )
    # ------------------------------------------------------------------
    # SECTION 3 -- Ranked candidates table
    # ------------------------------------------------------------------
    st.markdown("### Top Candidates")
    display_df = df.head(top_n).copy()

    # Show only what is in submission.csv — no enrichment from other files
    cols_to_show = ["candidate_id", "score", "rank", "reasoning"]
    st.dataframe(
        display_df[cols_to_show],
        use_container_width=True,
        hide_index=True,
    )

    # ------------------------------------------------------------------
    # SECTION 4 -- Per-candidate detail panels
    # ------------------------------------------------------------------
    if show_details and candidates_lookup:
        st.markdown("### Candidate Detail View")

        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            candidate_options = display_df["candidate_id"].tolist()
            selected_cid_dropdown = st.selectbox("Select candidate from dropdown", candidate_options)
        with col_sel2:
            search_cid_input = st.text_input("Or search by candidate ID directly", placeholder="e.g. CAND_0004989").strip()

        # Determine which candidate ID to use
        selected_cid = selected_cid_dropdown
        if search_cid_input:
            if search_cid_input in candidates_lookup:
                selected_cid = search_cid_input
            else:
                st.error(f"Candidate ID '{search_cid_input}' not found in candidates list.")

        record = candidates_lookup.get(selected_cid, {})
        profile = record.get("profile", {})
        signals = record.get("redrob_signals", {})
        skills = record.get("skills", [])
        career = record.get("career_history", [])

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Profile**")
            st.write(f"Name: {profile.get('anonymized_name', 'N/A')}")
            st.write(
                f"Title: {profile.get('current_title', 'N/A')} "
                f"@ {profile.get('current_company', 'N/A')}"
            )
            st.write(f"Location: {profile.get('location', 'N/A')}")
            st.write(f"Industry: {profile.get('current_industry', 'N/A')}")
            st.write(f"Experience: {profile.get('years_of_experience', 'N/A')}y")

            st.markdown("**Availability Signals**")
            st.write(
                f"Response Rate: "
                f"{fmt_sentinel(signals.get('recruiter_response_rate', -1))}"
            )
            st.write(
                f"Notice Period: {signals.get('notice_period_days', 'N/A')}d"
            )
            st.write(
                f"Open to Work: {signals.get('open_to_work_flag', 'N/A')}"
            )
            st.write(
                f"Profile Completeness: "
                f"{signals.get('profile_completeness_score', 'N/A')}%"
            )
            st.write(
                f"Interview Completion: "
                f"{fmt_sentinel(signals.get('interview_completion_rate', -1))}"
            )
            st.write(
                f"GitHub Activity: "
                f"{fmt_sentinel(signals.get('github_activity_score', -1))}"
            )

        with col_b:
            # Check if this candidate is ranked in the submission CSV
            matching_rows = df[df["candidate_id"] == selected_cid]
            if not matching_rows.empty:
                row = matching_rows.iloc[0]
                final_score_str = f"{row['score']:.4f}"
                rank_str = f"{row['rank']} / {config.TOP_K}"
                reasoning_str = row["reasoning"]
            else:
                final_score_str = "Not ranked (Not in Top 100)"
                rank_str = "N/A"
                reasoning_str = "This candidate is not present in the top-100 ranked candidates of the submission CSV."

            st.markdown("**Score Breakdown**")
            st.write(f"Final Score: {final_score_str}")
            st.write(f"Rank: {rank_str}")

            st.markdown("**Reasoning**")
            st.info(reasoning_str)

            st.markdown("**Top Skills**")
            if skills:
                skills_sorted = sorted(
                    skills,
                    key=lambda x: x.get("endorsements", 0),
                    reverse=True,
                )[:5]
                skills_df = pd.DataFrame(
                    [
                        {
                            "Skill": s["name"],
                            "Proficiency": s.get("proficiency", "N/A"),
                            "Months": s.get("duration_months", 0),
                            "Endorsements": s.get("endorsements", 0),
                        }
                        for s in skills_sorted
                    ]
                )
                st.dataframe(
                    skills_df, hide_index=True, use_container_width=True
                )
            else:
                st.write("No skills data available")

            st.markdown("**Career History**")
            if career:
                career_df = pd.DataFrame(
                    [
                        {
                            "Company": c.get("company", "N/A"),
                            "Title": c.get("title", "N/A"),
                            "Duration (mo)": c.get("duration_months", 0),
                            "Industry": c.get("industry", "N/A"),
                            "Current": "Yes" if c.get("is_current") else "No",
                        }
                        for c in career
                    ]
                )
                st.dataframe(
                    career_df, hide_index=True, use_container_width=True
                )
            else:
                st.write("No career history available")

    elif show_details and not candidates_lookup:
        st.info(
            "Upload candidates.jsonl in the sidebar to enable detail panels."
        )

    # ------------------------------------------------------------------
    # SECTION 5 -- Download button
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Export")
    st.download_button(
        label="Download submission.csv",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="submission.csv",
        mime="text/csv",
    )

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    st.markdown("---")
    st.caption("TalentRank AI -- Redrob Challenge Submission")


if __name__ == "__main__":
    main()
