"""Test the HNSW skeleton — level assignment distribution."""

from collections import Counter
from hnsw import HNSWIndex


def test_level_distribution():
    """Generate 10,000 levels and check the distribution matches theory."""
    index = HNSWIndex(M=16, seed=42)

    levels = [index._random_level() for _ in range(10_000)]
    counts = Counter(levels)

    print(f"Level distribution for M=16, 10,000 samples:\n")
    print(f"{'Level':<8}{'Count':<10}{'Actual %':<12}{'Expected %':<12}")
    print("-" * 42)

    # Expected: P(level = L) = (1 - 1/M) * (1/M)^L
    # Simplified: P(level >= L) = (1/M)^L
    M = 16
    for level in sorted(counts.keys()):
        actual_pct = counts[level] / 10_000 * 100
        # P(level = L) ≈ P(level >= L) - P(level >= L+1) = (1/M)^L - (1/M)^(L+1)
        expected_pct = ((1/M)**level - (1/M)**(level+1)) * 100
        print(f"{level:<8}{counts[level]:<10}{actual_pct:<12.1f}{expected_pct:<12.1f}")

    print(f"\nMax level seen: {max(levels)}")
    print(f"Nodes on layer 0: all 10,000 (everyone lives here)")
    print(f"Nodes that REACH layer 1+: {sum(1 for l in levels if l >= 1)}")
    print(f"Nodes that REACH layer 2+: {sum(1 for l in levels if l >= 2)}")

    # Sanity: most should be level 0
    assert counts[0] > 8000, "Expected >80% at level 0"
    print("\n✓ Distribution looks exponential")


def test_data_structure():
    """Verify the skeleton holds together."""
    index = HNSWIndex(M=16, ef_construction=200, metric="l2", seed=42)

    print(f"\nEmpty index: {index}")
    assert len(index) == 0
    assert index.entry_point is None
    assert index.max_level == -1
    print("✓ Empty index initialised correctly")


if __name__ == "__main__":
    test_level_distribution()
    test_data_structure()
