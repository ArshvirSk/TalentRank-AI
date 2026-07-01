"""
Tests for src.retrieval (Member 2's modules).

Covers:
  - jd_parser.py: JD parsing to structured profile
  - embed.py: embedding generation
  - faiss_index.py: FAISS index build & query
"""

from pathlib import Path

import numpy as np
import pytest

from src.retrieval.jd_parser import parse_job_description, JDProfile


# ---- JD Parser tests ----

class TestJDParser:
    """Tests for jd_parser.parse_job_description."""

    @pytest.mark.skip(reason="Not yet implemented — TODO(Member 2)")
    def test_parse_returns_jd_profile(self, tmp_path):
        """Parsing a valid .docx should return a JDProfile instance."""
        # TODO(Member 2): create a minimal test .docx fixture
        jd_path = tmp_path / "test_jd.docx"
        profile = parse_job_description(jd_path)
        assert isinstance(profile, JDProfile)

    @pytest.mark.skip(reason="Not yet implemented — TODO(Member 2)")
    def test_must_have_skills_populated(self, tmp_path):
        """Must-have skills should be a non-empty list."""
        jd_path = tmp_path / "test_jd.docx"
        profile = parse_job_description(jd_path)
        assert len(profile.must_have_skills) > 0

    def test_jd_profile_text_properties(self):
        """JDProfile text concatenation properties should work."""
        profile = JDProfile(
            must_have_skills=["Python", "PyTorch"],
            nice_to_have_skills=["Kubernetes"],
            disqualifiers=["no ML experience"],
        )
        assert "Python" in profile.must_have_text
        assert "Kubernetes" in profile.nice_to_have_text
        assert "no ML" in profile.disqualifier_text


# ---- Embedding tests ----

class TestEmbedding:
    """Tests for embed.embed_texts."""

    @pytest.mark.skip(reason="Not yet implemented — TODO(Member 2)")
    def test_embed_returns_correct_shape(self):
        """Embedding output should have shape (N, D)."""
        texts = ["hello world", "test sentence"]
        embeddings = embed_texts(texts)
        assert embeddings.shape == (2, 384)  # bge-small-en-v1.5

    @pytest.mark.skip(reason="Not yet implemented — TODO(Member 2)")
    def test_embed_deterministic(self):
        """Same input should produce the same embeddings."""
        texts = ["deterministic test"]
        e1 = embed_texts(texts)
        e2 = embed_texts(texts)
        np.testing.assert_array_almost_equal(e1, e2)


# ---- FAISS tests ----

class TestFAISS:
    """Tests for faiss_index build & query."""

    @pytest.mark.skip(reason="Not yet implemented — TODO(Member 2)")
    def test_build_and_query_roundtrip(self, tmp_path):
        """Build an index and query it — top result should be self-match."""
        # Create dummy embeddings
        embeddings = np.random.randn(100, 384).astype(np.float32)
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
        assert indices[0, 0] == 0  # self should be top match
