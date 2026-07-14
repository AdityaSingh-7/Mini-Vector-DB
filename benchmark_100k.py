"""
Benchmark MY HNSW on SIFT-100K (first 100K vectors from SIFT-1M).

This is the "put up or shut up" test: does our recall still match
hnswlib at a 10× larger scale than our previous 10K benchmark?

Expected runtime: ~7 min build + ~2 min queries = ~10 min total.

Requires: sift-128-euclidean.hdf5 (already downloaded)
"""

import time
import numpy as np
import h5py

from hnsw import HNSWIndex
from brute_force import BruteForceIndex
import hnswlib

# ──────────────────────────────────────────────────────────────────────
# Load first 100K from SIFT-1M
# ──────────────────────────────────────────────────────────────────────

print("Loading SIFT-100K subset...")
f = h5py.File("sift-128-euclidean.hdf5", "r")
data = np.array(f["train"][:100000])
queries = np.array(f["test"][:200])
N, DIM = data.shape
K = 10
print(f"  Data: {N:,} × {DIM}D, Queries: {len(queries)}, k={K}\n")

# ──────────────────────────────────────────────────────────────────────
# Ground truth (brute force on 100K)
# ──────────────────────────────────────────────────────────────────────

print("Computing ground truth (brute force)...")
brute = BruteForceIndex(metric="l2")
brute.add_batch(data)

ground_truth = []
for q in queries:
    res = brute.query(q, k=K)
    ground_truth.append(set(idx for idx, _ in res))
print("  Done.\n")


def compute_recall(results_list, gt, k):
    return np.mean([len(r & t) / k for r, t in zip(results_list, gt)])


# ──────────────────────────────────────────────────────────────────────
# Build MY HNSW on 100K
# ──────────────────────────────────────────────────────────────────────

print("Building My HNSW (100K vectors)...")
t0 = time.perf_counter()
my_hnsw = HNSWIndex(M=16, ef_construction=200, metric="l2", seed=42)
for i, v in enumerate(data):
    my_hnsw.insert(v)
    if (i + 1) % 10000 == 0:
        elapsed = time.perf_counter() - t0
        rate = (i + 1) / elapsed
        eta = (N - i - 1) / rate
        print(f"  {i+1:,}/{N:,} ({rate:.0f}/s, ETA: {eta/60:.1f} min)")

build_time = time.perf_counter() - t0
print(f"  Build complete: {build_time:.1f}s ({N/build_time:.0f} inserts/sec)\n")

# ──────────────────────────────────────────────────────────────────────
# Benchmark MY HNSW
# ──────────────────────────────────────────────────────────────────────

EF_VALUES = [10, 20, 50, 100, 200]

print(f"My HNSW results:")
print(f"  {'ef':<8}{'Recall@10':<12}{'QPS':<10}")
print(f"  {'-'*30}")
my_points = []
for ef in EF_VALUES:
    results_list = []
    start = time.perf_counter()
    for q in queries:
        res = my_hnsw.query(q, k=K, ef_search=ef)
        results_list.append(set(idx for idx, _ in res))
    elapsed = time.perf_counter() - start
    qps = len(queries) / elapsed
    recall = compute_recall(results_list, ground_truth, K)
    my_points.append((ef, recall, qps))
    print(f"  {ef:<8}{recall:<12.3f}{qps:<10.0f}")

# ──────────────────────────────────────────────────────────────────────
# Benchmark hnswlib for comparison
# ──────────────────────────────────────────────────────────────────────

print(f"\nhnswlib results:")
hnsw_lib = hnswlib.Index(space="l2", dim=DIM)
hnsw_lib.init_index(max_elements=N, M=16, ef_construction=200, random_seed=42)
hnsw_lib.add_items(data)

print(f"  {'ef':<8}{'Recall@10':<12}{'QPS':<10}")
print(f"  {'-'*30}")
lib_points = []
for ef in EF_VALUES:
    hnsw_lib.set_ef(ef)
    start = time.perf_counter()
    labels, _ = hnsw_lib.knn_query(queries, k=K)
    elapsed = time.perf_counter() - start
    results_list = [set(row) for row in labels]
    qps = len(queries) / elapsed
    recall = compute_recall(results_list, ground_truth, K)
    lib_points.append((ef, recall, qps))
    print(f"  {ef:<8}{recall:<12.3f}{qps:<10.0f}")

# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"SIFT-100K COMPARISON")
print(f"{'='*60}")
print(f"{'ef':<6}{'My Recall':<12}{'My QPS':<10}{'hnswlib Recall':<16}{'hnswlib QPS':<12}{'Speed gap':<10}")
print(f"{'-'*60}")
for (ef, mr, mq), (_, lr, lq) in zip(my_points, lib_points):
    gap = lq / mq if mq > 0 else 0
    print(f"{ef:<6}{mr:<12.3f}{mq:<10.0f}{lr:<16.3f}{lq:<12.0f}{gap:.0f}×")
print(f"{'='*60}")
print(f"\nIf recalls match (within ~2%), the bullet upgrades to '100K vectors'.")
print(f"If gap is still ~70×, the claim holds at scale.")
