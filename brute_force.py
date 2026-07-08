"""
v0 — Brute-Force K-Nearest Neighbours Index

The simplest possible vector search: compare the query against EVERY stored
vector, sort by distance, return the top-k. O(N·d) per query.

This serves two purposes:
  1. Baseline to understand what we're improving on.
  2. Ground-truth "oracle" — since it's exact, we use it to measure how
     accurate our approximate index (HNSW) is later.
"""

import numpy as np
from typing import Literal


class BruteForceIndex:
    """Exact nearest-neighbour search by exhaustive scan."""

    def __init__(self, metric: Literal["l2", "cosine"] = "l2"):
        """
        Args:
            metric: Distance function to use.
                - "l2": Euclidean distance (smaller = more similar)
                - "cosine": 1 - cosine_similarity (smaller = more similar)
        """
        self.metric = metric
        self.vectors = []  # will become a numpy array on first query
        self._dirty = True  # tracks whether we need to rebuild the matrix

    def add(self, vector: np.ndarray):
        """Add a single vector to the index."""
        self.vectors.append(np.array(vector, dtype=np.float32))
        self._dirty = True

    def add_batch(self, vectors: np.ndarray):
        """Add multiple vectors at once. vectors shape: (n, dim)"""
        for v in vectors:
            self.vectors.append(np.array(v, dtype=np.float32))
        self._dirty = True

    def _build_matrix(self):
        """Stack all vectors into a single numpy matrix for fast computation."""
        if self._dirty and self.vectors:
            self._matrix = np.vstack(self.vectors)  # shape: (N, dim)
            if self.metric == "cosine":
                # Pre-normalize so cosine distance = 1 - dot product
                norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
                # Avoid division by zero
                norms = np.maximum(norms, 1e-10)
                self._matrix_normed = self._matrix / norms
            self._dirty = False

    def query(self, query_vector: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
        """
        Find the k nearest neighbours to query_vector.

        Returns:
            List of (index, distance) tuples, sorted nearest-first.
        """
        self._build_matrix()
        q = np.array(query_vector, dtype=np.float32)

        if self.metric == "l2":
            # Euclidean distance: ||a - b||^2 (we skip the sqrt — ranking is the same)
            # Efficient expansion: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
            diffs = self._matrix - q  # (N, dim)
            distances = np.sum(diffs ** 2, axis=1)  # (N,)

        elif self.metric == "cosine":
            # Cosine distance = 1 - cosine_similarity
            q_norm = q / max(np.linalg.norm(q), 1e-10)
            similarities = self._matrix_normed @ q_norm  # (N,)
            distances = 1.0 - similarities

        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        # Get the k smallest distances
        # argpartition is O(N) vs argsort's O(N log N) — faster for large N
        if k >= len(distances):
            top_k_indices = np.argsort(distances)
        else:
            top_k_indices = np.argpartition(distances, k)[:k]
            # Sort just the top-k by distance
            top_k_indices = top_k_indices[np.argsort(distances[top_k_indices])]

        return [(int(idx), float(distances[idx])) for idx in top_k_indices]

    def __len__(self):
        return len(self.vectors)
