"""
Benchmark: My HNSW vs hnswlib vs FAISS on Structured Dataset

Uses a 50-cluster synthetic dataset (10K vectors, 128D) that mimics
real-world ANN benchmark characteristics — clustered data with varying
density, which is harder than uniform random vectors.

Produces a recall@10 vs QPS chart.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from brute_force import BruteForceIndex
from hnsw import HNSWIndex
import hnswlib
import faiss

# ──────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────

data = np.load("benchmark_data.npy")
queries = np.load("benchmark_queries.npy")
N, DIM = data.shape
N_QUERIES = len(queries)
K = 10

print(f"Dataset: {N:,} vectors, {DIM}D, {N_QUERIES} queries, k={K}")
print(f"Structure: 50 clusters, varying density (harder than random)\n")

EF_VALUES = [10, 20, 30, 50, 75, 100, 150, 200, 300, 400]


# ──────────────────────────────────────────────────────────────────────
# Ground truth
# ──────────────────────────────────────────────────────────────────────

print("Computing ground truth (brute-force)...")
brute = BruteForceIndex(metric="l2")
brute.add_batch(data)

ground_truth = []
for q in queries:
    results = brute.query(q, k=K)
    ground_truth.append(set(idx for idx, _ in results))

start = time.perf_counter()
for q in queries:
    brute.query(q, k=K)
brute_time = time.perf_counter() - start
brute_qps = N_QUERIES / brute_time
print(f"  Brute-force: {brute_qps:.0f} QPS\n")


def compute_recall(results_list, ground_truth, k):
    recalls = []
    for result_ids, true_ids in zip(results_list, ground_truth):
        recalls.append(len(result_ids & true_ids) / k)
    return np.mean(recalls)


# ──────────────────────────────────────────────────────────────────────
# My HNSW
# ──────────────────────────────────────────────────────────────────────

print("Building My HNSW...")
t0 = time.perf_counter()
my_hnsw = HNSWIndex(M=16, ef_construction=200, metric="l2", seed=42)
for v in data:
    my_hnsw.insert(v)
build_time = time.perf_counter() - t0
print(f"  Build: {build_time:.1f}s ({N/build_time:.0f} inserts/sec)")

my_points = []
print(f"\n  {'ef':<8}{'Recall@10':<12}{'QPS':<10}")
print(f"  {'-'*30}")
for ef in EF_VALUES:
    results_list = []
    start = time.perf_counter()
    for q in queries:
        res = my_hnsw.query(q, k=K, ef_search=ef)
        results_list.append(set(idx for idx, _ in res))
    elapsed = time.perf_counter() - start
    qps = N_QUERIES / elapsed
    recall = compute_recall(results_list, ground_truth, K)
    my_points.append((qps, recall))
    print(f"  {ef:<8}{recall:<12.3f}{qps:<10.0f}")


# ──────────────────────────────────────────────────────────────────────
# hnswlib
# ──────────────────────────────────────────────────────────────────────

print("\nBuilding hnswlib...")
hnsw_lib = hnswlib.Index(space="l2", dim=DIM)
hnsw_lib.init_index(max_elements=N, M=16, ef_construction=200, random_seed=42)
hnsw_lib.add_items(data)

lib_points = []
print(f"\n  {'ef':<8}{'Recall@10':<12}{'QPS':<10}")
print(f"  {'-'*30}")
for ef in EF_VALUES:
    hnsw_lib.set_ef(ef)
    start = time.perf_counter()
    labels, _ = hnsw_lib.knn_query(queries, k=K)
    elapsed = time.perf_counter() - start
    results_list = [set(row) for row in labels]
    qps = N_QUERIES / elapsed
    recall = compute_recall(results_list, ground_truth, K)
    lib_points.append((qps, recall))
    print(f"  {ef:<8}{recall:<12.3f}{qps:<10.0f}")


# ──────────────────────────────────────────────────────────────────────
# FAISS
# ──────────────────────────────────────────────────────────────────────

print("\nBuilding FAISS HNSW...")
faiss_index = faiss.IndexHNSWFlat(DIM, 16)
faiss_index.hnsw.efConstruction = 200
faiss_index.add(data)

faiss_points = []
print(f"\n  {'ef':<8}{'Recall@10':<12}{'QPS':<10}")
print(f"  {'-'*30}")
for ef in EF_VALUES:
    faiss_index.hnsw.efSearch = ef
    start = time.perf_counter()
    _, labels = faiss_index.search(queries, K)
    elapsed = time.perf_counter() - start
    results_list = [set(row) for row in labels]
    qps = N_QUERIES / elapsed
    recall = compute_recall(results_list, ground_truth, K)
    faiss_points.append((qps, recall))
    print(f"  {ef:<8}{recall:<12.3f}{qps:<10.0f}")


# ──────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────

print("\nGenerating chart...")
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.plot([p[0] for p in my_points], [p[1] for p in my_points],
        'o-', color='#e74c3c', linewidth=2, markersize=6, label='My HNSW (Python)')
ax.plot([p[0] for p in lib_points], [p[1] for p in lib_points],
        's-', color='#3498db', linewidth=2, markersize=6, label='hnswlib (C++)')
ax.plot([p[0] for p in faiss_points], [p[1] for p in faiss_points],
        '^-', color='#2ecc71', linewidth=2, markersize=6, label='FAISS HNSW (C++)')

ax.axvline(x=brute_qps, color='gray', linestyle='--', alpha=0.7)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
ax.annotate(f'Brute-force\n({brute_qps:.0f} QPS, exact)',
            xy=(brute_qps, 0.70), fontsize=9, color='gray', ha='center')

ax.set_xlabel('Queries per Second (QPS) →', fontsize=12)
ax.set_ylabel('Recall@10 →', fontsize=12)
ax.set_title(f'Recall vs Throughput — {N:,} vectors, {DIM}D, 50 clusters, k={K}', fontsize=14)
ax.set_xscale('log')
ax.set_ylim(0.65, 1.02)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)

ax.annotate('Same algorithm,\ndifferent language\n(Python vs C++)',
            xy=(my_points[4][0], my_points[4][1]),
            xytext=(my_points[4][0]*0.3, my_points[4][1]-0.05),
            fontsize=9, color='#e74c3c', alpha=0.8,
            arrowprops=dict(arrowstyle='->', color='#e74c3c', alpha=0.5))

plt.tight_layout()
plt.savefig('benchmark_recall_vs_qps.png', dpi=150, bbox_inches='tight')
print("  Saved: benchmark_recall_vs_qps.png")

# Summary
print(f"\n{'='*60}")
print(f"SUMMARY at ~95% recall:")
print(f"{'='*60}")
for name, points in [("My HNSW", my_points), ("hnswlib", lib_points), ("FAISS", faiss_points)]:
    for qps, recall in points:
        if recall >= 0.95:
            print(f"  {name:<15} → {qps:>8.0f} QPS  (recall={recall:.3f})")
            break
print(f"  {'Brute-force':<15} → {brute_qps:>8.0f} QPS  (recall=1.000)")
print(f"{'='*60}")
