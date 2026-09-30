"""
Stage 1 — JD Understanding.

Owner: Member 2 (implemented post-hackathon by Arshvir — see docs/OWNERSHIP.md)

Parse the job description (.docx) into a structured requirement profile
with separate text blocks for:

  - must-have skills
  - nice-to-have skills
  - explicit disqualifiers ("do not want")
  - ideal-candidate profile (free-text description)
  - location/logistics preferences

The parser is deliberately boring: keyword-routed section splitting +
bullet/bold-run harvesting + skill-phrase extraction.  It runs once per
JD before embedding, so runtime cost is irrelevant; correctness and
debuggability beat ML cleverness here.  The canonical JD used in the
hackathon is already hand-decomposed in src/retrieval/build_texts.py
(JD_MUST_HAVE / JD_NICE_TO_HAVE / JD_DISQUALIFIER) — this module makes the
pipeline re-runnable on arbitrary JDs instead of only that one.
"""

from __future__ import annotations

import re
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


# ---------------------------------------------------------------------------
# Section routing
# ---------------------------------------------------------------------------

# Ordered (regex, target_field) rules — first matching rule wins.
# 'nice' patterns are checked before 'must' so that e.g. a "Nice to have:
# strong Python" bullet lands in nice_to_have_skills, not must_have_skills.
_SECTION_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(disqualifier|do[- ]not[- ]want|don't want|exclude|red[- ]flag|deal[- ]?breaker|anti[- ]requirement)\b", re.I), "disqualifiers"),
    (re.compile(r"\b(nice[- ]to[- ]have|preferred|plus|bonus|good[- ]to[- ]have|desirable)\b", re.I), "nice"),
    (re.compile(r"\b(must[- ]have|required|requirements|essential|minimum qualification|basic qualification|you have|you bring|what you('ll| will) need|qualifications)\b", re.I), "must"),
]

_BULLET_RE = re.compile(r"^\s*[\u2022\u2023\u25aa\u25cf\-–—*·>]+\s*")
_NUMBERED_RE = re.compile(r"^\s*\d+[.)]\s*")

# Skill-ish phrase: capitalized tokens, acronyms, or known tech punctuation.
# Used to pull items out of prose lines; intentionally conservative.
_SKILL_TOKEN_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9+#.]*(?:\s+(?:of|for|and)\s+[A-Z][A-Za-z0-9+#.]*)?\b"
)

_LOCATION_RE = re.compile(
    r"\b(?:locat\w*|based in|remote|hybrid|on[- ]site|relocat\w*|visa)\b[:\s]*(.*)",
    re.I,
)


def _detect_section(line: str, current: str) -> str:
    """Return the target field for this line, or the continued section."""
    for pattern, target in _SECTION_RULES:
        if pattern.search(line):
            return target
    # A short standalone line (<= 60 chars, no sentence punctuation) that
    # doesn't match any rule is treated as a section header for its own sake
    # only if it ends with ':' — otherwise it's body text of the current
    # section.
    return current


def _clean_line(line: str) -> str:
    """Strip bullets, numbering, and stray whitespace."""
    line = _BULLET_RE.sub("", line)
    line = _NUMBERED_RE.sub("", line)
    return line.strip().rstrip(";,")


_LEADING_HEADER_RE = re.compile(
    r"^(must[- ]have|required|requirements?|nice[- ]to[- ]have|preferred|plus|bonus|good[- ]to[- ]have|disqualifiers?|do[- ]not[- ]want|don't want|exclusions?)\s*[:\-]\s*",
    re.I,
)


def _extract_skills_from_line(line: str) -> list[str]:
    """Extract skill-ish phrases from one line of section body text.

    Strategy: if the line looks like a comma/semicolon-separated list
    (common in JDs), split on separators and keep fragments 1–40 chars.
    Otherwise, harvest capitalized/technical tokens from prose; if the line
    has none (e.g. an all-lowercase bullet like 'pure academic research'),
    keep the whole line as one phrase — for disqualifier sentences the full
    phrase is exactly what we want to embed.
    """
    line = _LEADING_HEADER_RE.sub("", line)

    if "," in line or ";" in line or "/" in line[:40]:
        fragments = re.split(r"[,;]| (?:and|or) ", line)
        skills = []
        for frag in fragments:
            frag = frag.strip(" .:•-")
            if 1 < len(frag) <= 40 and not frag.lower().startswith(("years", "experience")):
                skills.append(frag)
        if skills:
            return skills

    # Prose fallback: capitalized tokens (skips sentence-initial false hits
    # poorly, but JD bullet sentences are usually skill-dense)
    candidates = []
    for match in _SKILL_TOKEN_RE.finditer(line):
        token = match.group(0).strip()
        if 1 < len(token) <= 40:
            candidates.append(token)
    if candidates:
        return candidates

    # Whole-line fallback for lowercase bullets/sentences
    stripped = line.strip(" .:•-")
    if 1 < len(stripped) <= 80:
        return [stripped]
    return []


def parse_job_description(jd_path: str | Path) -> JDProfile:
    """
    Parse a .docx job description into a structured ``JDProfile``.

    Parameters
    ----------
    jd_path : str or Path
        Path to the job_description.docx file.  (.txt is accepted too.)

    Returns
    -------
    JDProfile
        Structured requirement profile.

    Raises
    ------
    FileNotFoundError
        If ``jd_path`` does not exist.
    ValueError
        If no readable text could be extracted from the document.

    Notes
    -----
    Implementation:
      1. Read paragraphs (and table cells) via python-docx.
      2. Route each line into must-have / nice-to-have / disqualifier /
         ideal-candidate buckets using the keyword rules in _SECTION_RULES.
      3. Bullets become list items; prose lines contribute extracted
         skill phrases; the full text is always preserved in raw_text.
      4. Missing sections degrade gracefully (empty lists).
    """
    path = Path(jd_path)
    if not path.exists():
        raise FileNotFoundError(f"Job description file not found: {path}")

    if path.suffix.lower() in {".txt", ".md"}:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    else:
        try:
            from docx import Document
        except ImportError as exc:
            raise ImportError(
                "python-docx is required to parse .docx job descriptions: "
                "pip install python-docx"
            ) from exc

        document = Document(str(path))
        text_parts: list[str] = []
        for paragraph in document.paragraphs:
            text_parts.append(paragraph.text)
        # JDs frequently put requirements in tables — harvest cells too.
        for table in document.tables:
            for row in table.rows:
                for cell in row.cells:
                    for paragraph in cell.paragraphs:
                        text_parts.append(paragraph.text)
        lines = "\n".join(text_parts).splitlines()

    profile = JDProfile()
    profile.raw_text = "\n".join(lines)

    current = "ideal"  # leading text (before any header) = description
    for raw_line in lines:
        line = _clean_line(raw_line)
        if not line:
            continue

        current = _detect_section(line, current)
        target = current

        # Header lines (end with ':') route the section but contribute no items
        if line.endswith(":"):
            continue

        if target == "ideal":
            # Accumulate free-text description; also try location capture
            profile.ideal_candidate_text = (profile.ideal_candidate_text + " " + line).strip()
            loc_match = _LOCATION_RE.search(line)
            if loc_match and not profile.location_preferences:
                profile.location_preferences = loc_match.group(1).strip()[:200]
            continue

        if target == "location":
            profile.location_preferences = (profile.location_preferences + " " + line).strip()[:300]
            continue

        bucket: list[str] = {
            "must": profile.must_have_skills,
            "nice": profile.nice_to_have_skills,
            "disqualifiers": profile.disqualifiers,
        }[target]  # type: ignore[assignment]

        extracted = _extract_skills_from_line(line)
        if extracted:
            bucket.extend(extracted)

    # Deduplicate preserving order, lowercase-normalizing near-duplicates
    def _dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            key = item.lower().strip()
            if key and key not in seen:
                seen.add(key)
                out.append(item.strip())
        return out

    profile.must_have_skills = _dedupe(profile.must_have_skills)
    profile.nice_to_have_skills = _dedupe(profile.nice_to_have_skills)
    profile.disqualifiers = _dedupe(profile.disqualifiers)

    if not profile.raw_text.strip():
        raise ValueError(f"No readable text extracted from {path}")

    return profile


def load_default_jd_profile() -> JDProfile:
    """
    Return the canonical hackathon JD as a JDProfile without reading the
    .docx — the hand-decomposed texts in build_texts.py are the source of
    truth for the competition's fixed job description.
    """
    from src.retrieval.build_texts import (
        JD_DISQUALIFIER,
        JD_MUST_HAVE,
        JD_NICE_TO_HAVE,
    )

    def _split(text: str) -> list[str]:
        fragments = []
        for chunk in re.split(r"[.\n]", text):
            chunk = chunk.strip()
            if chunk:
                fragments.append(chunk)
        return fragments

    return JDProfile(
        must_have_skills=_split(JD_MUST_HAVE),
        nice_to_have_skills=_split(JD_NICE_TO_HAVE),
        disqualifiers=_split(JD_DISQUALIFIER),
        ideal_candidate_text="AI engineer in India for a Pune/Noida product-company ML role.",
        location_preferences="India; willing to relocate to Pune or Noida",
    )
