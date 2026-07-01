"""
person_b/benchmark_models.py

Run this FIRST to pick the embedding model.
Tests BGE-small, E5-small, MiniLM on 1000 texts and extrapolates to 100k.

Usage: python benchmark_models.py
"""

import json, time
import numpy as np
from sentence_transformers import SentenceTransformer
from build_texts import build_candidate_text

# Load sample candidates (use sample_candidates.json from the data bundle)
with open("../data/sample_candidates.json") as f:
    data = json.load(f)

texts = [build_candidate_text(c) for c in data]
texts_1k = (texts * 20)[:1000]   # 1000 texts to benchmark

word_counts = [len(t.split()) for t in texts]
print(f"Text stats — min: {min(word_counts)} max: {max(word_counts)} "
      f"median: {int(np.median(word_counts))} words")
print()

models = [
    ("BAAI/bge-small-en-v1.5",  "Best quality/speed, designed for asymmetric retrieval"),
    ("intfloat/e5-small-v2",    "Good quality, needs query:/passage: prefixes"),
    ("all-MiniLM-L6-v2",        "Fastest, slightly weaker on nuance"),
]

results = []
for model_name, note in models:
    try:
        model = SentenceTransformer(model_name)
        # Warm-up
        model.encode(["warm up"], show_progress_bar=False)
        # Timed run
        t = time.time()
        embeddings = model.encode(
            texts_1k,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        elapsed = time.time() - t
        per_100k_min = elapsed / 1000 * 100000 / 60
        dim = embeddings.shape[1]
        results.append((model_name, elapsed, per_100k_min, dim, note))
        print(f"✓ {model_name}")
        print(f"  {elapsed:.1f}s for 1k texts → ~{per_100k_min:.1f} min for 100k")
        print(f"  Embedding dim: {dim} | Note: {note}")
        print()
    except Exception as e:
        print(f"✗ {model_name}: {e}\n")

# Recommendation
if results:
    best = min(results, key=lambda x: x[2])  # fastest
    print(f"Recommendation: {best[0]} (~{best[2]:.1f} min for 100k)")
    print("If quality matters more than speed, prefer BAAI/bge-small-en-v1.5")
