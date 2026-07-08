"""Quick sanity check for BruteForceIndex."""

import numpy as np
from brute_force import BruteForceIndex


def test_basic():
    """5 vectors in 3D, query should find the nearest one."""
    index = BruteForceIndex(metric="l2")

    # Plant 5 known vectors
    vectors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.5, 0.5, 0.5],
    ])
    index.add_batch(vectors)

    # Query near [0.5, 0.5, 0.5] — vector #4 should be the nearest
    results = index.query(np.array([0.6, 0.6, 0.6]), k=3)

    print("Query: [0.6, 0.6, 0.6]")
    print("Top-3 nearest neighbours:")
    for rank, (idx, dist) in enumerate(results, 1):
        print(f"  #{rank}: vector[{idx}] = {vectors[idx]} — distance = {dist:.4f}")

    assert results[0][0] == 4, f"Expected index 4, got {results[0][0]}"
    print("\n✓ Nearest neighbour is correct (index 4)\n")


def test_cosine():
    """Cosine metric: direction matters, not magnitude."""
    index = BruteForceIndex(metric="cosine")

    vectors = np.array([
        [1.0, 0.0],    # pointing right
        [0.0, 1.0],    # pointing up
        [1.0, 1.0],    # 45 degrees
        [10.0, 10.0],  # also 45 degrees but 10x longer!
    ])
    index.add_batch(vectors)

    # Query at 45 degrees — vectors 2 and 3 should tie (same direction)
    results = index.query(np.array([3.0, 3.0]), k=4)

    print("Query: [3.0, 3.0] (45 degrees)")
    print("Cosine distances:")
    for idx, dist in results:
        print(f"  vector[{idx}] = {vectors[idx]} — cosine_dist = {dist:.6f}")

    # Vectors 2 and 3 should both have cosine distance ≈ 0 (same direction)
    assert abs(results[0][1]) < 1e-5, "Expected near-zero cosine distance"
    assert abs(results[1][1]) < 1e-5, "Expected near-zero cosine distance"
    print("\n✓ Cosine metric ignores magnitude correctly\n")


def test_scale():
    """10,000 vectors — how fast is brute force?"""
    import time

    dim = 128
    n_vectors = 10_000
    index = BruteForceIndex(metric="l2")

    # Random vectors
    np.random.seed(42)
    data = np.random.randn(n_vectors, dim).astype(np.float32)
    index.add_batch(data)

    # Time 100 queries
    queries = np.random.randn(100, dim).astype(np.float32)
    start = time.perf_counter()
    for q in queries:
        index.query(q, k=10)
    elapsed = time.perf_counter() - start

    qps = 100 / elapsed
    avg_ms = (elapsed / 100) * 1000
    print(f"10,000 vectors × 128 dims:")
    print(f"  100 queries in {elapsed:.3f}s")
    print(f"  {qps:.0f} queries/sec, {avg_ms:.2f} ms/query")
    print(f"\n✓ Brute force works but {'⚠️  slow' if qps < 500 else 'fast enough at this scale'}\n")


if __name__ == "__main__":
    test_basic()
    test_cosine()
    test_scale()
