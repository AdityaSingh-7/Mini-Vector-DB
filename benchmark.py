"""
Benchmark: My HNSW vs hnswlib vs FAISS vs Brute-Force

Produces a recall@10 vs QPS chart on 10,000 random vectors (128D).
This is the resume artifact — shows our implementation achieves
comparable recall curves, just at lower throughput (pure Python vs C++).
"""

import time
import numpy as np
import matplotlib.pyplot as plt

# Our implementations
from brute_force import BruteForceIndex
from hnsw import HNSWIndex

# Industrial libraries
import hnswlib
import faiss


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

N = 10_000        # number of vectors in the index
DIM = 128         # dimensionality
N_QUERIES = 200   # number of queries to average over
K = 10            # neighbours to retrieve
SEED = 42

# ef_search values to sweep (the recall-vs-speed dial)
EF_VALUES = [10, 20, 30, 50, 75, 100, 150, 200, 300, 400]


# ──────────────────────────────────────────────────────────────────────
# Generate data
# ──────────────────────────────────────────────────────────────────────

print(f"Generating {N} vectors, {DIM}D...")
np.random.seed(SEED)
data = np.random.randn(N, DIM).astype(np.float32)
queries = np.random.randn(N_QUERIES, DIM).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
# Ground truth (brute-force)
# ──────────────────────────────────────────────────────────────────────

print("Computing ground truth (brute-force)...")
brute = BruteForceIndex(metric="l2")
brute.add_batch(data)

ground_truth = []
for q in queries:
    results = brute.query(q, k=K)
    ground_truth.append(set(idx for idx, _ in results))

# Also time brute-force
start = time.perf_counter()
for q in queries:
    brute.query(q, k=K)
brute_time = time.perf_counter() - start
brute_qps = N_QUERIES / brute_time
print(f"  Brute-force: {brute_qps:.0f} QPS ({(brute_time/N_QUERIES)*1000:.2f} ms/query)")


def compute_recall(results_list, ground_truth, k):
    """Average recall@k across all queries."""
    recalls = []
    for result_ids, true_ids in zip(results_list, ground_truth):
        recalls.append(len(result_ids & true_ids) / k)
    return np.mean(recalls)


# ──────────────────────────────────────────────────────────────────────
# Benchmark: My HNSW
# ──────────────────────────────────────────────────────────────────────

print("\nBuilding My HNSW index...")
t0 = time.perf_counter()
my_hnsw = HNSWIndex(M=16, ef_construction=200, metric="l2", seed=SEED)
for v in data:
    my_hnsw.insert(v)
my_build_time = time.perf_counter() - t0
print(f"  Build time: {my_build_time:.1f}s ({N/my_build_time:.0f} inserts/sec)")
print(f"  {my_hnsw}")

my_hnsw_points = []  # (qps, recall)
print(f"\n  {'ef_search':<12}{'Recall@10':<12}{'QPS':<10}{'ms/query':<10}")
print(f"  {'-'*42}")

for ef in EF_VALUES:
    results_list = []
    start = time.perf_counter()
    for q in queries:
        res = my_hnsw.query(q, k=K, ef_search=ef)
        results_list.append(set(idx for idx, _ in res))
    elapsed = time.perf_counter() - start

    qps = N_QUERIES / elapsed
    recall = compute_recall(results_list, ground_truth, K)
    my_hnsw_points.append((qps, recall))
    print(f"  {ef:<12}{recall:<12.3f}{qps:<10.0f}{(elapsed/N_QUERIES)*1000:<10.2f}")


# ──────────────────────────────────────────────────────────────────────
# Benchmark: hnswlib
# ──────────────────────────────────────────────────────────────────────

print("\nBuilding hnswlib index...")
hnsw_lib = hnswlib.Index(space="l2", dim=DIM)
hnsw_lib.init_index(max_elements=N, M=16, ef_construction=200, random_seed=SEED)

t0 = time.perf_counter()
hnsw_lib.add_items(data)
lib_build_time = time.perf_counter() - t0
print(f"  Build time: {lib_build_time:.2f}s ({N/lib_build_time:.0f} inserts/sec)")

hnswlib_points = []  # (qps, recall)
print(f"\n  {'ef_search':<12}{'Recall@10':<12}{'QPS':<10}{'ms/query':<10}")
print(f"  {'-'*42}")

for ef in EF_VALUES:
    hnsw_lib.set_ef(ef)
    start = time.perf_counter()
    labels, distances = hnsw_lib.knn_query(queries, k=K)
    elapsed = time.perf_counter() - start

    results_list = [set(row) for row in labels]
    qps = N_QUERIES / elapsed
    recall = compute_recall(results_list, ground_truth, K)
    hnswlib_points.append((qps, recall))
    print(f"  {ef:<12}{recall:<12.3f}{qps:<10.0f}{(elapsed/N_QUERIES)*1000:<10.2f}")


# ──────────────────────────────────────────────────────────────────────
# Benchmark: FAISS (HNSW)
# ──────────────────────────────────────────────────────────────────────

print("\nBuilding FAISS HNSW index...")
faiss_index = faiss.IndexHNSWFlat(DIM, 16)  # M=16
faiss_index.hnsw.efConstruction = 200

t0 = time.perf_counter()
faiss_index.add(data)
faiss_build_time = time.perf_counter() - t0
print(f"  Build time: {faiss_build_time:.2f}s ({N/faiss_build_time:.0f} inserts/sec)")

faiss_points = []  # (qps, recall)
print(f"\n  {'ef_search':<12}{'Recall@10':<12}{'QPS':<10}{'ms/query':<10}")
print(f"  {'-'*42}")

for ef in EF_VALUES:
    faiss_index.hnsw.efSearch = ef
    start = time.perf_counter()
    distances, labels = faiss_index.search(queries, K)
    elapsed = time.perf_counter() - start

    results_list = [set(row) for row in labels]
    qps = N_QUERIES / elapsed
    recall = compute_recall(results_list, ground_truth, K)
    faiss_points.append((qps, recall))
    print(f"  {ef:<12}{recall:<12.3f}{qps:<10.0f}{(elapsed/N_QUERIES)*1000:<10.2f}")


# ──────────────────────────────────────────────────────────────────────
# Plot: Recall@10 vs QPS
# ──────────────────────────────────────────────────────────────────────

print("\nGenerating chart...")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Plot each method
my_qps = [p[0] for p in my_hnsw_points]
my_recall = [p[1] for p in my_hnsw_points]
ax.plot(my_qps, my_recall, 'o-', color='#e74c3c', linewidth=2, markersize=6,
        label='My HNSW (Python)')

lib_qps = [p[0] for p in hnswlib_points]
lib_recall = [p[1] for p in hnswlib_points]
ax.plot(lib_qps, lib_recall, 's-', color='#3498db', linewidth=2, markersize=6,
        label='hnswlib (C++)')

faiss_qps = [p[0] for p in faiss_points]
faiss_recall = [p[1] for p in faiss_points]
ax.plot(faiss_qps, faiss_recall, '^-', color='#2ecc71', linewidth=2, markersize=6,
        label='FAISS HNSW (C++)')

# Brute-force reference line
ax.axvline(x=brute_qps, color='gray', linestyle='--', alpha=0.7)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
ax.annotate(f'Brute-force\n({brute_qps:.0f} QPS, exact)',
            xy=(brute_qps, 0.75), fontsize=9, color='gray', ha='center')

# Labels and styling
ax.set_xlabel('Queries per Second (QPS) →', fontsize=12)
ax.set_ylabel('Recall@10 →', fontsize=12)
ax.set_title(f'Recall vs Throughput — {N:,} vectors, {DIM}D, k={K}', fontsize=14)
ax.set_xscale('log')
ax.set_ylim(0.7, 1.02)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)

# Add annotation explaining the gap
ax.annotate('Same algorithm,\ndifferent language\n(Python vs C++)',
            xy=(my_qps[4], my_recall[4]),
            xytext=(my_qps[4]*0.3, my_recall[4]-0.05),
            fontsize=9, color='#e74c3c', alpha=0.8,
            arrowprops=dict(arrowstyle='->', color='#e74c3c', alpha=0.5))

plt.tight_layout()
plt.savefig('benchmark_recall_vs_qps.png', dpi=150, bbox_inches='tight')
print(f"  Saved: benchmark_recall_vs_qps.png")

# Also print a summary table
print(f"\n{'='*60}")
print(f"SUMMARY — at ~95% recall:")
print(f"{'='*60}")
# Find ef that gives ~95% recall for each
for name, points in [("My HNSW", my_hnsw_points), ("hnswlib", hnswlib_points), ("FAISS", faiss_points)]:
    for qps, recall in points:
        if recall >= 0.95:
            print(f"  {name:<15} → {qps:>8.0f} QPS  (recall={recall:.3f})")
            break
print(f"  {'Brute-force':<15} → {brute_qps:>8.0f} QPS  (recall=1.000)")
print(f"{'='*60}")
