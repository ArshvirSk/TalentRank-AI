"""
Tests for src.retrieval (Member 2's modules).

Covers:
  - jd_parser.py: JD parsing to structured profile
  - faiss_index.py: FAISS index build & query, numpy-backend equivalence

The numpy retrieval backend is exercised by tests/test_ranking.py; here we
verify the FAISS path agrees with it (backend equivalence) and that the JD
parser produces sane structured output.
"""

from pathlib import Path

import numpy as np
import pytest

from src.retrieval.jd_parser import parse_job_description, JDProfile

faiss = pytest.importorskip("faiss")  # module-level: FAISS tests need faiss-cpu

from src.retrieval.faiss_index import build_faiss_index, normalize_rows, query_faiss_index  # noqa: E402


# ---- JD Parser tests ----

class TestJDParser:
    """Tests for jd_parser.parse_job_description."""

    def _write_docx(self, tmp_path: Path, paragraphs: list[str]) -> Path:
        docx = pytest.importorskip("docx")
        document = docx.Document()
        for text in paragraphs:
            document.add_paragraph(text)
        path = tmp_path / "test_jd.docx"
        document.save(str(path))
        return path

    def test_parse_returns_jd_profile(self, tmp_path):
        jd_path = self._write_docx(tmp_path, [
            "Senior ML Engineer",
            "Must have: strong Python, PyTorch, and production ML experience",
            "Nice to have: Kubernetes, MLflow",
            "Disqualifiers: consulting-only background",
        ])
        profile = parse_job_description(jd_path)
        assert isinstance(profile, JDProfile)
        assert profile.raw_text

    def test_must_have_skills_populated(self, tmp_path):
        jd_path = self._write_docx(tmp_path, [
            "Requirements:",
            "- Python",
            "- PyTorch",
            "- Docker",
        ])
        profile = parse_job_description(jd_path)
        assert len(profile.must_have_skills) > 0

    def test_section_routing(self, tmp_path):
        """Bullets route to the right bucket based on section headers."""
        jd_path = self._write_docx(tmp_path, [
            "Requirements:",
            "- Python",
            "- SQL",
            "Nice to have:",
            "- Kubernetes",
            "Do not want:",
            "- pure academic research",
        ])
        profile = parse_job_description(jd_path)
        joined_must = " ".join(profile.must_have_skills).lower()
        joined_nice = " ".join(profile.nice_to_have_skills).lower()
        joined_disq = " ".join(profile.disqualifiers).lower()
        assert "python" in joined_must and "sql" in joined_must
        assert "kubernetes" in joined_nice
        assert "academic" in joined_disq
        # Cross-contamination guard: Kubernetes must not leak into must-have
        assert "kubernetes" not in joined_must

    def test_missing_sections_degrade(self, tmp_path):
        """A JD with no recognized headers still parses (everything ideal)."""
        jd_path = self._write_docx(tmp_path, ["Just some text about a role."])
        profile = parse_job_description(jd_path)
        assert profile.raw_text
        assert isinstance(profile.must_have_skills, list)

    def test_txt_input_supported(self, tmp_path):
        path = tmp_path / "jd.txt"
        path.write_text("Requirements:\n- Python\n- FastAPI\n", encoding="utf-8")
        profile = parse_job_description(path)
        assert len(profile.must_have_skills) >= 2

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_job_description(tmp_path / "nope.docx")

    def test_default_profile_loader(self):
        from src.retrieval.jd_parser import load_default_jd_profile

        profile = load_default_jd_profile()
        assert profile.must_have_skills and profile.nice_to_have_skills and profile.disqualifiers
        assert "Python" in profile.must_have_text


# ---- FAISS tests ----

class TestFAISS:
    """Tests for faiss_index build & query."""

    def test_build_and_query_roundtrip(self, tmp_path):
        """Build an index and query it — top result should be self-match."""
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((100, 384)).astype(np.float32)
        emb_path = tmp_path / "test_embeddings.npy"
        np.save(emb_path, embeddings)

        index_path = tmp_path / "test.faiss"
        build_faiss_index(embeddings_path=emb_path, index_output_path=index_path)

        # Query with the first embedding
        query = embeddings[:1]
        distances, indices = query_faiss_index(
            query_vectors=query, top_k=5, index_path=index_path
        )
        assert indices.shape == (1, 5)
        assert distances.shape == (1, 5)
        assert indices[0, 0] == 0  # self should be top match
        assert distances[0, 0] == pytest.approx(1.0, abs=1e-5)  # cosine self-sim

    def test_unnormalized_embeddings_handled(self, tmp_path):
        """Non-normalized inputs are normalized internally (cosine, not dot)."""
        rng = np.random.default_rng(0)
        embeddings = rng.standard_normal((50, 16)).astype(np.float32) * 5.0
        index_path = build_faiss_index(embeddings=embeddings,
                                       index_output_path=tmp_path / "t.faiss")
        distances, indices = query_faiss_index(
            embeddings[:1], top_k=3, index_path=index_path
        )
        assert indices[0, 0] == 0
        assert distances[0, 0] == pytest.approx(1.0, abs=1e-5)

    def test_equivalence_with_numpy_backend(self, tmp_path):
        """FAISS IndexFlatIP and the numpy matmul produce identical top-k."""
        rng = np.random.default_rng(7)
        embeddings = normalize_rows(rng.standard_normal((500, 64)).astype(np.float32))
        queries = normalize_rows(rng.standard_normal((3, 64)).astype(np.float32))

        index_path = build_faiss_index(embeddings=embeddings,
                                       index_output_path=tmp_path / "t.faiss")
        d_faiss, i_faiss = query_faiss_index(queries, top_k=20, index_path=index_path)

        sims = embeddings @ queries.T  # (N, 3)
        for qi in range(3):
            np_top = np.argsort(sims[:, qi])[::-1][:20]
            np.testing.assert_array_equal(i_faiss[qi], np_top)
            np.testing.assert_allclose(d_faiss[qi], sims[np_top, qi], atol=1e-5)

    def test_query_missing_index_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            query_faiss_index(np.zeros((1, 8), dtype=np.float32),
                              top_k=5, index_path=tmp_path / "absent.faiss")
