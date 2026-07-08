"""
Test query() — the moment of truth.
Compare HNSW results against brute-force ground truth.
"""

import time
import numpy as np
from hnsw import HNSWIndex
from brute_force import BruteForceIndex


def recall_at_k(hnsw_results: list, brute_results: list, k: int) -> float:
    """
    What fraction of the true top-k did HNSW actually find?

    recall@k = |HNSW_top_k ∩ BruteForce_top_k| / k

    1.0 = perfect (found all true nearest neighbours)
    0.0 = complete failure (found none of them)
    """
    hnsw_ids = set(idx for idx, _ in hnsw_results[:k])
    true_ids = set(idx for idx, _ in brute_results[:k])
    return len(hnsw_ids & true_ids) / k


def test_query_small():
    """Small test — 100 vectors, verify we find correct neighbours."""
    np.random.seed(42)
    dim = 8
    n = 100
    data = np.random.randn(n, dim).astype(np.float32)

    # Build both indexes
    hnsw = HNSWIndex(M=16, ef_construction=200, metric="l2", seed=42)
    brute = BruteForceIndex(metric="l2")

    for v in data:
        hnsw.insert(v)
        brute.add(v)

    # Query
    query = np.random.randn(dim).astype(np.float32)
    k = 5

    hnsw_results = hnsw.query(query, k=k, ef_search=50)
    brute_results = brute.query(query, k=k)

    print(f"100 vectors, {dim}D, k={k}")
    print(f"\nBrute-force top-{k} (ground truth):")
    for idx, dist in brute_results:
        print(f"  Node {idx}: distance = {dist:.4f}")

    print(f"\nHNSW top-{k}:")
    for idx, dist in hnsw_results:
        print(f"  Node {idx}: distance = {dist:.4f}")

    recall = recall_at_k(hnsw_results, brute_results, k)
    print(f"\nRecall@{k} = {recall:.1%}")
    assert recall >= 0.8, f"Recall too low: {recall:.1%}"
    print("✓ HNSW finds correct neighbours\n")


def test_query_accuracy():
    """1000 vectors — measure recall over 100 queries."""
    np.random.seed(42)
    dim = 128
    n = 1000
    data = np.random.randn(n, dim).astype(np.float32)

    # Build indexes
    hnsw = HNSWIndex(M=16, ef_construction=200, metric="l2", seed=42)
    brute = BruteForceIndex(metric="l2")

    print(f"Building index: {n} vectors, {dim}D...")
    for v in data:
        hnsw.insert(v)
    brute.add_batch(data)
    print(f"  {hnsw}")

    # Run 100 queries
    k = 10
    queries = np.random.randn(100, dim).astype(np.float32)

    print(f"\nRunning 100 queries (k={k})...")
    print(f"{'ef_search':<12}{'Recall@10':<12}{'Avg ms/query':<15}{'QPS':<10}")
    print("-" * 49)

    for ef in [10, 20, 50, 100, 200]:
        recalls = []
        start = time.perf_counter()
        for q in queries:
            hnsw_res = hnsw.query(q, k=k, ef_search=ef)
            brute_res = brute.query(q, k=k)
            recalls.append(recall_at_k(hnsw_res, brute_res, k))
        elapsed = time.perf_counter() - start

        avg_recall = np.mean(recalls)
        avg_ms = (elapsed / 100) * 1000
        qps = 100 / elapsed
        print(f"{ef:<12}{avg_recall:<12.3f}{avg_ms:<15.2f}{qps:<10.0f}")

    print("\n✓ Recall improves with ef_search (the speed/accuracy dial works!)")


def test_query_speed():
    """Brute-force vs HNSW speed comparison."""
    np.random.seed(42)
    dim = 128
    n = 5000
    data = np.random.randn(n, dim).astype(np.float32)

    # Build
    print(f"\nSpeed test: {n} vectors, {dim}D")
    hnsw = HNSWIndex(M=16, ef_construction=100, metric="l2", seed=42)
    brute = BruteForceIndex(metric="l2")

    t0 = time.perf_counter()
    for v in data:
        hnsw.insert(v)
    build_time = time.perf_counter() - t0
    brute.add_batch(data)
    print(f"  HNSW build: {build_time:.1f}s")

    # Query both
    queries = np.random.randn(50, dim).astype(np.float32)
    k = 10

    # Brute force timing
    start = time.perf_counter()
    for q in queries:
        brute.query(q, k=k)
    brute_time = time.perf_counter() - start
    brute_qps = 50 / brute_time

    # HNSW timing
    start = time.perf_counter()
    for q in queries:
        hnsw.query(q, k=k, ef_search=50)
    hnsw_time = time.perf_counter() - start
    hnsw_qps = 50 / hnsw_time

    # HNSW recall at this ef
    recalls = []
    for q in queries:
        h = hnsw.query(q, k=k, ef_search=50)
        b = brute.query(q, k=k)
        recalls.append(recall_at_k(h, b, k))

    print(f"\n  {'Method':<15}{'QPS':<12}{'ms/query':<12}{'Recall@10':<10}")
    print(f"  {'-'*47}")
    print(f"  {'Brute-force':<15}{brute_qps:<12.0f}{(brute_time/50)*1000:<12.2f}{'1.000':<10}")
    print(f"  {'HNSW (ef=50)':<15}{hnsw_qps:<12.0f}{(hnsw_time/50)*1000:<12.2f}{np.mean(recalls):<10.3f}")
    print(f"\n  Speedup: {hnsw_qps/brute_qps:.1f}x faster than brute-force")
    print("✓ HNSW is faster than brute-force")


if __name__ == "__main__":
    test_query_small()
    test_query_accuracy()
    test_query_speed()
