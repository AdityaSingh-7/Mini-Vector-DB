"""Test insert — build an HNSW graph and inspect it."""

import numpy as np
from hnsw import HNSWIndex


def test_insert_first_node():
    """First node should just set the entry point, no edges."""
    index = HNSWIndex(M=4, ef_construction=50, seed=42)
    index.insert(np.array([1.0, 2.0, 3.0]))

    print("After inserting 1 node:")
    print(f"  vectors: {len(index.vectors)}")
    print(f"  entry_point: {index.entry_point}")
    print(f"  max_level: {index.max_level}")
    print(f"  level of node 0: {index.levels[0]}")

    assert len(index) == 1
    assert index.entry_point == 0
    print("✓ First node inserted correctly\n")


def test_insert_small_graph():
    """Insert 10 nodes and check the graph looks reasonable."""
    index = HNSWIndex(M=4, ef_construction=50, metric="l2", seed=42)

    # 10 vectors in 2D — easy to visualise
    np.random.seed(42)
    vectors = np.random.randn(10, 2).astype(np.float32)

    for v in vectors:
        index.insert(v)

    print(f"After inserting 10 nodes: {index}")
    print(f"  Entry point: node {index.entry_point} (level {index.levels[index.entry_point]})")
    print(f"  Levels: {index.levels}")
    print()

    # Check Layer 0 — all nodes should be here
    print("Layer 0 (all nodes):")
    for node_id in sorted(index.graphs[0].keys()):
        friends = index.graphs[0][node_id]
        print(f"  Node {node_id}: {len(friends)} friends → {friends}")

    # Verify all 10 nodes are on layer 0
    assert len(index.graphs[0]) == 10, f"Expected 10 nodes on layer 0, got {len(index.graphs[0])}"

    # Check edge structure — note: edges are NOT always bidirectional
    # After trimming, some edges become one-way. This is expected in HNSW.
    # (If A→B exists but B's friends got trimmed, B→A might be gone.)
    bidir_count = 0
    total_edges = 0
    for node_id, friends in index.graphs[0].items():
        for friend_id in friends:
            total_edges += 1
            if node_id in index.graphs[0][friend_id]:
                bidir_count += 1

    bidir_pct = bidir_count / total_edges * 100
    print(f"\n  Edge stats: {total_edges} total edges, {bidir_pct:.0f}% bidirectional")
    print("  (Some one-way edges are expected after neighbour trimming)")
    print("✓ Graph structure looks healthy")

    # Check no node exceeds M_max0 = 2*M = 8 edges on layer 0
    for node_id, friends in index.graphs[0].items():
        assert len(friends) <= index.M_max0, \
            f"Node {node_id} has {len(friends)} friends, max is {index.M_max0}"
    print(f"✓ No node exceeds M_max0={index.M_max0} edges on layer 0")

    # Check higher layers
    if len(index.graphs) > 1:
        print(f"\nHigher layers:")
        for layer in range(1, len(index.graphs)):
            nodes = list(index.graphs[layer].keys())
            print(f"  Layer {layer}: {len(nodes)} nodes → {nodes}")


def test_insert_larger():
    """Insert 1000 nodes — check structure and timing."""
    import time

    index = HNSWIndex(M=16, ef_construction=200, metric="l2", seed=42)

    np.random.seed(42)
    vectors = np.random.randn(1000, 128).astype(np.float32)

    start = time.perf_counter()
    for v in vectors:
        index.insert(v)
    elapsed = time.perf_counter() - start

    print(f"\n1000 vectors × 128 dims:")
    print(f"  {index}")
    print(f"  Build time: {elapsed:.2f}s ({1000/elapsed:.0f} inserts/sec)")
    print(f"  Entry point: node {index.entry_point} (level {index.levels[index.entry_point]})")

    # Layer stats
    for layer in range(len(index.graphs)):
        n_nodes = len(index.graphs[layer])
        avg_edges = np.mean([len(f) for f in index.graphs[layer].values()])
        print(f"  Layer {layer}: {n_nodes} nodes, avg {avg_edges:.1f} edges/node")

    print(f"\n✓ Large insert completed successfully")


if __name__ == "__main__":
    test_insert_first_node()
    test_insert_small_graph()
    test_insert_larger()
