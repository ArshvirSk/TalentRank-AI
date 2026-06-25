"""
TalentRank AI — Streamlit Sandbox Demo.

Owner: Member 4

A small-sample interactive demo per submission spec section 10.5.
Allows judges to explore a subset of ranked candidates, view score
breakdowns, and understand the pipeline's reasoning.

Usage:
    streamlit run app/streamlit_app.py
"""

import streamlit as st


def main():
    """Launch the Streamlit sandbox demo."""
    st.set_page_config(
        page_title="TalentRank AI — Candidate Ranking Demo",
        page_icon="🏆",
        layout="wide",
    )

    st.title("🏆 TalentRank AI")
    st.subheader("Intelligent Candidate Discovery & Ranking")

    st.markdown(
        """
        This demo showcases the ranking pipeline's output on a small sample.
        Upload a submission CSV or load the default to explore ranked candidates.
        """
    )

    # --- Sidebar: configuration ---
    st.sidebar.header("Configuration")
    csv_file = st.sidebar.file_uploader(
        "Upload submission.csv", type=["csv"]
    )

    # TODO(Member 4): implement the following sections
    #
    # 1. Load submission CSV (uploaded or default from config.SUBMISSION_PATH)
    # 2. Display top-K candidates in a sortable table
    # 3. Per-candidate detail view:
    #    a. Score breakdown bar chart
    #    b. Reasoning text
    #    c. Key signals (production evidence, honeypot flags, etc.)
    # 4. Score distribution histogram
    # 5. JD requirement coverage heatmap

    st.info(
        "🚧 Demo under construction. The ranking pipeline scaffold is in "
        "place — run `python -m src.ranking.rank` once implemented to "
        "generate a submission CSV for this demo."
    )

    st.markdown("---")
    st.caption("TalentRank AI — Redrob Challenge Submission")


if __name__ == "__main__":
    main()
