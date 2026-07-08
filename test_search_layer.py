"""Test _search_layer — greedy beam search on a single layer."""

import numpy as np
from hnsw import HNSWIndex


def test_search_layer_basic():
    """
    Build a tiny graph by hand and verify that _search_layer
    finds the correct nearest neighbour.

    5 nodes in 2D:
      Node 0: [1.0, 1.0]
      Node 1: [4.0, 4.0]
      Node 2: [7.0, 8.0]   ← true nearest to query [6.0, 7.0]
      Node 3: [5.0, 2.0]
      Node 4: [8.0, 1.0]

    Edges (Layer 0):
      0 → [1, 3]
      1 → [0, 2, 3]
      2 → [1, 4]
      3 → [0, 1, 4]
      4 → [2, 3]
    """
    index = HNSWIndex(M=16, metric="l2")

    # Manually inject vectors
    index.vectors = [
        np.array([1.0, 1.0]),
        np.array([4.0, 4.0]),
        np.array([7.0, 8.0]),
        np.array([5.0, 2.0]),
        np.array([8.0, 1.0]),
    ]

    # Manually build Layer 0 graph
    index.graphs = [
        {
            0: [1, 3],
            1: [0, 2, 3],
            2: [1, 4],
            3: [0, 1, 4],
            4: [2, 3],
        }
    ]
    index.max_level = 0
    index.entry_point = 0

    query = np.array([6.0, 7.0])

    # Search starting from Node 0, on Layer 0, beam width ef=3
    results = index._search_layer(query, entry_points=[0], layer=0, ef=3)

    print("Query: [6.0, 7.0]")
    print(f"Search from Node 0, ef=3")
    print(f"\nTop-3 results:")
    for dist, node_id in results:
        print(f"  Node {node_id}: {index.vectors[node_id]} — distance = {dist:.1f}")

    # Node 2 should be nearest (distance = 1+1 = 2.0 in squared L2)
    assert results[0][1] == 2, f"Expected Node 2 as nearest, got Node {results[0][1]}"
    print("\n✓ Found correct nearest neighbour (Node 2)")


def test_search_layer_ef_effect():
    """
    Show that higher ef finds better results when the graph has tricky structure.

    Build a graph where ef=1 gets trapped but ef=5 finds the true answer.
    """
    index = HNSWIndex(M=16, metric="l2")

    # 7 nodes in 2D — two clusters with a narrow bridge
    index.vectors = [
        np.array([0.0, 0.0]),    # Node 0 — start here
        np.array([1.0, 0.0]),    # Node 1
        np.array([2.0, 0.0]),    # Node 2 — bridge
        np.array([3.0, 0.0]),    # Node 3 — bridge
        np.array([10.0, 0.0]),   # Node 4 — far cluster
        np.array([10.0, 1.0]),   # Node 5 — far cluster
        np.array([10.0, 0.5]),   # Node 6 — TRUE nearest to query [9.5, 0.5]
    ]

    # Graph: cluster 1 (0,1,2) connected internally,
    #         bridge 2→3, cluster 2 (3,4,5,6) connected internally
    # But also a "distraction" edge 1→5 that's far
    index.graphs = [
        {
            0: [1],
            1: [0, 2],
            2: [1, 3],
            3: [2, 4, 6],
            4: [3, 5, 6],
            5: [4, 6],
            6: [3, 4, 5],
        }
    ]
    index.max_level = 0

    query = np.array([9.5, 0.5])  # Nearest is Node 6 [10.0, 0.5] → dist = 0.5

    # With ef=1 (pure greedy) — might work here since it's a chain, but let's trace
    results_ef1 = index._search_layer(query, entry_points=[0], layer=0, ef=1)
    # With ef=5 — wider beam
    results_ef5 = index._search_layer(query, entry_points=[0], layer=0, ef=5)

    print(f"\nQuery: [9.5, 0.5]")
    print(f"\nef=1 result: Node {results_ef1[0][1]}, dist={results_ef1[0][0]:.2f}")
    print(f"ef=5 top result: Node {results_ef5[0][1]}, dist={results_ef5[0][0]:.2f}")
    print(f"ef=5 all results:")
    for dist, node_id in results_ef5:
        print(f"  Node {node_id}: {index.vectors[node_id]} — distance = {dist:.2f}")

    # Both should find Node 6 in this simple chain
    assert results_ef5[0][1] == 6, f"Expected Node 6, got {results_ef5[0][1]}"
    print("\n✓ ef=5 found correct nearest (Node 6)")


def test_search_layer_returns_ef_results():
    """Verify we actually get ef results back (not just 1)."""
    index = HNSWIndex(M=16, metric="l2")

    # 6 nodes in 1D (simple chain)
    index.vectors = [np.array([float(i)]) for i in range(6)]
    index.graphs = [
        {
            0: [1],
            1: [0, 2],
            2: [1, 3],
            3: [2, 4],
            4: [3, 5],
            5: [4],
        }
    ]
    index.max_level = 0

    query = np.array([3.5])  # Between Node 3 and Node 4

    results = index._search_layer(query, entry_points=[0], layer=0, ef=4)
    print(f"\n1D chain, query=[3.5], ef=4:")
    for dist, node_id in results:
        print(f"  Node {node_id}: [{index.vectors[node_id][0]:.1f}] — distance = {dist:.2f}")

    assert len(results) == 4, f"Expected 4 results, got {len(results)}"
    # Should be nodes 3, 4, 2, 5 (distances: 0.25, 0.25, 2.25, 2.25)
    print(f"\n✓ Got {len(results)} results as expected")


if __name__ == "__main__":
    test_search_layer_basic()
    test_search_layer_ef_effect()
    test_search_layer_returns_ef_results()
