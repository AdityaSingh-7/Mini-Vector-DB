"""
Benchmark on SIFT-1M (1,000,000 vectors, 128D) — the standard ANN benchmark.

Uses the HDF5 file from ann-benchmarks.com which includes pre-computed ground truth.
Download: http://ann-benchmarks.com/sift-128-euclidean.hdf5 (~501 MB)
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import h5py

import hnswlib
import faiss


# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────

MAX_QUERIES = 500
K = 10
EF_VALUES = [10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500]


# ──────────────────────────────────────────────────────────────────────
# Load SIFT-1M
# ──────────────────────────────────────────────────────────────────────

HDF5_PATH = os.path.join(os.path.dirname(__file__), "sift-128-euclidean.hdf5")

if not os.path.exists(HDF5_PATH):
    print("ERROR: sift-128-euclidean.hdf5 not found!")
    print("Download from: http://ann-benchmarks.com/sift-128-euclidean.hdf5 (~501 MB)")
    exit(1)

print("Loading SIFT-1M dataset...")
f = h5py.File(HDF5_PATH, 'r')
data = np.array(f['train'])          # (1,000,000, 128)
queries = np.array(f['test'][:MAX_QUERIES])   # (500, 128)
gt_neighbors = np.array(f['neighbors'][:MAX_QUERIES, :K])  # (500, 10) — ground truth

N, DIM = data.shape
N_QUERIES = len(queries)

print(f"  Data: {N:,} × {DIM}D")
print(f"  Queries: {N_QUERIES}")
print(f"  k={K}\n")

ground_truth = [set(gt_neighbors[i]) for i in range(N_QUERIES)]


def compute_recall(results_list, ground_truth, k):
    recalls = []
    for result_ids, true_ids in zip(results_list, ground_truth):
        recalls.append(len(result_ids & true_ids) / k)
    return np.mean(recalls)


# ──────────────────────────────────────────────────────────────────────
# hnswlib
# ──────────────────────────────────────────────────────────────────────

print("Building hnswlib index (M=16, ef_construction=200)...")
t0 = time.perf_counter()
hnsw_lib = hnswlib.Index(space="l2", dim=DIM)
hnsw_lib.init_index(max_elements=N, M=16, ef_construction=200, random_seed=42)
hnsw_lib.add_items(data, num_threads=4)
lib_build = time.perf_counter() - t0
print(f"  Build time: {lib_build:.1f}s ({N/lib_build:.0f} inserts/sec)")

lib_points = []
print(f"\n  {'ef':<8}{'Recall@10':<12}{'QPS':<10}{'Latency':<10}")
print(f"  {'-'*40}")
for ef in EF_VALUES:
    hnsw_lib.set_ef(ef)
    start = time.perf_counter()
    labels, _ = hnsw_lib.knn_query(queries, k=K)
    elapsed = time.perf_counter() - start
    results_list = [set(row) for row in labels]
    qps = N_QUERIES / elapsed
    recall = compute_recall(results_list, ground_truth, K)
    latency_ms = (elapsed / N_QUERIES) * 1000
    lib_points.append((qps, recall))
    print(f"  {ef:<8}{recall:<12.3f}{qps:<10.0f}{latency_ms:.2f}ms")


# ──────────────────────────────────────────────────────────────────────
# FAISS
# ──────────────────────────────────────────────────────────────────────

print("\nBuilding FAISS HNSW index (M=16, ef_construction=200)...")
t0 = time.perf_counter()
faiss_index = faiss.IndexHNSWFlat(DIM, 16)
faiss_index.hnsw.efConstruction = 200
faiss_index.add(data)
faiss_build = time.perf_counter() - t0
print(f"  Build time: {faiss_build:.1f}s ({N/faiss_build:.0f} inserts/sec)")

faiss_points = []
print(f"\n  {'ef':<8}{'Recall@10':<12}{'QPS':<10}{'Latency':<10}")
print(f"  {'-'*40}")
for ef in EF_VALUES:
    faiss_index.hnsw.efSearch = ef
    start = time.perf_counter()
    _, labels = faiss_index.search(queries, K)
    elapsed = time.perf_counter() - start
    results_list = [set(row) for row in labels]
    qps = N_QUERIES / elapsed
    recall = compute_recall(results_list, ground_truth, K)
    latency_ms = (elapsed / N_QUERIES) * 1000
    faiss_points.append((qps, recall))
    print(f"  {ef:<8}{recall:<12.3f}{qps:<10.0f}{latency_ms:.2f}ms")


# ──────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────

print("\nGenerating chart...")
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.plot([p[0] for p in lib_points], [p[1] for p in lib_points],
        's-', color='#3498db', linewidth=2, markersize=6, label='hnswlib (C++)')
ax.plot([p[0] for p in faiss_points], [p[1] for p in faiss_points],
        '^-', color='#2ecc71', linewidth=2, markersize=6, label='FAISS HNSW (C++)')

ax.set_xlabel('Queries per Second (QPS) →', fontsize=12)
ax.set_ylabel('Recall@10 →', fontsize=12)
ax.set_title(f'Recall vs Throughput — SIFT-1M ({N:,} vectors, {DIM}D, k={K})', fontsize=14)
ax.set_xscale('log')
ax.set_ylim(0.65, 1.02)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('benchmark_sift1m.png', dpi=150, bbox_inches='tight')
print("  Saved: benchmark_sift1m.png")

# Summary
print(f"\n{'='*60}")
print(f"SIFT-1M BENCHMARK at ~95% recall:")
print(f"{'='*60}")
for name, points in [("hnswlib", lib_points), ("FAISS", faiss_points)]:
    for qps, recall in points:
        if recall >= 0.95:
            print(f"  {name:<15} → {qps:>8.0f} QPS  (recall={recall:.3f})")
            break
print(f"{'='*60}")
print(f"\nNote: Python HNSW not benchmarked on 1M (would take hours to build).")
print(f"The 10K benchmark confirms the algorithm matches — the gap is purely")
print(f"Python vs C++ throughput.")
