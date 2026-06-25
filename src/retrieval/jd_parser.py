"""
Stage 1 — JD Understanding.

Owner: Member 2

Parse the job description (.docx) into a structured requirement profile
with separate text blocks for:

  - must-have skills
  - nice-to-have skills
  - explicit disqualifiers ("do not want")
  - ideal-candidate profile (free-text description)
  - location/logistics preferences
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class JDProfile:
    """
    Structured representation of a parsed job description.

    Each text block is a plain string (or list of strings) ready to be
    embedded by Stage 3.  Downstream stages should never re-read the .docx.
    """
    must_have_skills: list[str] = field(default_factory=list)
    nice_to_have_skills: list[str] = field(default_factory=list)
    disqualifiers: list[str] = field(default_factory=list)
    ideal_candidate_text: str = ""
    location_preferences: str = ""
    raw_text: str = ""

    # TODO(Member 2): add any additional JD fields useful for scoring

    @property
    def must_have_text(self) -> str:
        """Concatenated must-have skills as a single string for embedding."""
        return " ".join(self.must_have_skills)

    @property
    def nice_to_have_text(self) -> str:
        """Concatenated nice-to-have skills as a single string for embedding."""
        return " ".join(self.nice_to_have_skills)

    @property
    def disqualifier_text(self) -> str:
        """Concatenated disqualifier terms as a single string for embedding."""
        return " ".join(self.disqualifiers)


def parse_job_description(jd_path: str | Path) -> JDProfile:
    """
    Parse a .docx job description into a structured ``JDProfile``.

    Parameters
    ----------
    jd_path : str or Path
        Path to the job_description.docx file.

    Returns
    -------
    JDProfile
        Structured requirement profile.

    Raises
    ------
    FileNotFoundError
        If ``jd_path`` does not exist.
    NotImplementedError
        Scaffold — not yet implemented.

    Notes
    -----
    Implementation checklist (Member 2):
      1. Use python-docx to read all paragraphs.
      2. Identify section headers (bold/styled or keyword-based) to split
         the document into must-have, nice-to-have, disqualifier, and
         ideal-candidate sections.
      3. Normalize bullet points and whitespace.
      4. Handle edge cases: missing sections, merged cells if the JD uses
         tables, etc.
    """
    # TODO(Member 2): implement JD parsing with python-docx
    raise NotImplementedError(
        "parse_job_description: JD parsing not yet implemented"
    )
