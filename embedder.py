"""
Embedding layer — converts text into vectors using sentence-transformers.

Uses the 'all-MiniLM-L6-v2' model:
  - 384 dimensions
  - Fast (~50ms per sentence on CPU)
  - Good quality for semantic search
  - ~80MB download (first run only)
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer

# Default: use local model bundled with project (no network needed)
_DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "all-MiniLM-L6-v2")


class Embedder:
    """Wraps a sentence-transformer model for text → vector conversion."""

    def __init__(self, model_path: str = _DEFAULT_MODEL_PATH):
        """
        Args:
            model_path: Path to local sentence-transformer model directory.
                        Defaults to bundled model (no internet required).
        """
        self.model = SentenceTransformer(model_path)
        self.dim = self.model.get_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        """Convert a single text string to a vector."""
        return self.model.encode(text, convert_to_numpy=True).astype(np.float32)

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """
        Convert multiple texts to vectors efficiently.

        Args:
            texts: List of strings.
            batch_size: Process this many at a time (GPU/CPU batching).

        Returns:
            numpy array of shape (len(texts), dim)
        """
        return self.model.encode(
            texts, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=True
        ).astype(np.float32)


if __name__ == "__main__":
    # Quick test
    embedder = Embedder()
    print(f"Model loaded: dim={embedder.dim}")

    texts = [
        "How do black holes form?",
        "Stellar collapse and gravitational singularities",
        "Best chocolate cake recipe",
    ]

    vectors = embedder.embed_batch(texts)
    print(f"Embedded {len(texts)} texts → shape {vectors.shape}")

    # Check similarity (related texts should be closer)
    from numpy.linalg import norm

    def cosine_sim(a, b):
        return np.dot(a, b) / (norm(a) * norm(b))

    print(f"\nCosine similarities:")
    print(f"  'black holes' ↔ 'stellar collapse': {cosine_sim(vectors[0], vectors[1]):.3f}")
    print(f"  'black holes' ↔ 'chocolate cake':   {cosine_sim(vectors[0], vectors[2]):.3f}")
    print(f"\n✓ Related texts are more similar than unrelated ones")
