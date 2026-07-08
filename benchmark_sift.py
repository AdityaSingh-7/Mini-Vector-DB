"""
How to benchmark on SIFT-10K (the standard ANN benchmark dataset).

Steps:
1. Download siftsmall.tar.gz from: http://corpus-texmex.irisa.fr/siftsmall.tar.gz
   (Alternative mirror: https://github.com/facebookresearch/faiss/wiki/Datasets)
   Size: ~7.5 MB

2. Extract it:
   tar -xzf siftsmall.tar.gz

   You'll get a folder 'siftsmall/' containing:
     siftsmall_base.fvecs      — 10,000 vectors (128D) to index
     siftsmall_query.fvecs     — 100 query vectors
     siftsmall_groundtruth.ivecs — 100 ground-truth nearest neighbours per query

3. Place the 'siftsmall/' folder inside ~/mini-vecdb/

4. Run this script:
   cd ~/mini-vecdb
   source .venv/bin/activate
   python benchmark_sift.py
"""

import os
import time
import struct
import numpy as np
import matplotlib.pyplot as plt

from brute_force import BruteForceIndex
from hnsw import HNSWIndex
import hnswlib
import faiss


# ──────────────────────────────────────────────────────────────────────
# Helper: read .fvecs and .ivecs format (standard texmex format)
# ──────────────────────────────────────────────────────────────────────

def read_fvecs(filename):
    """Read .fvecs file → numpy array of float32."""
    with open(filename, 'rb') as f:
        data = []
        while True:
            # Each vector: [dim (4 bytes int)] [dim floats (4 bytes each)]
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vec = struct.unpack(f'{dim}f', f.read(dim * 4))
            data.append(vec)
    return np.array(data, dtype=np.float32)


def read_ivecs(filename):
    """Read .ivecs file → numpy array of int32 (ground truth indices)."""
    with open(filename, 'rb') as f:
        data = []
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vec = struct.unpack(f'{dim}i', f.read(dim * 4))
            data.append(vec)
    return np.array(data, dtype=np.int32)


# ──────────────────────────────────────────────────────────────────────
# Load SIFT-10K
# ──────────────────────────────────────────────────────────────────────

SIFT_DIR = os.path.join(os.path.dirname(__file__), "siftsmall")

if not os.path.exists(SIFT_DIR):
    print("ERROR: 'siftsmall/' folder not found!")
    print()
    print("Download it from: http://corpus-texmex.irisa.fr/siftsmall.tar.gz")
    print("Then extract:     tar -xzf siftsmall.tar.gz")
    print("Place the 'siftsmall/' folder in this directory.")
    exit(1)

print("Loading SIFT-10K dataset...")
data = read_fvecs(os.path.join(SIFT_DIR, "siftsmall_base.fvecs"))
queries = read_fvecs(os.path.join(SIFT_DIR, "siftsmall_query.fvecs"))
gt = read_ivecs(os.path.join(SIFT_DIR, "siftsmall_groundtruth.ivecs"))

N, DIM = data.shape
N_QUERIES = len(queries)
K = 10

print(f"  Base vectors: {data.shape}")
print(f"  Queries: {queries.shape}")
print(f"  Ground truth: {gt.shape}")
print(f"  k={K}\n")

# Ground truth: take top-K for each query
ground_truth = [set(gt[i, :K]) for i in range(N_QUERIES)]

EF_VALUES = [10, 20, 30, 50, 75, 100, 150, 200, 300, 400]


def compute_recall(results_list, ground_truth, k):
    recalls = []
    for result_ids, true_ids in zip(results_list, ground_truth):
        recalls.append(len(result_ids & true_ids) / k)
    return np.mean(recalls)


# ──────────────────────────────────────────────────────────────────────
# Brute-force baseline
# ──────────────────────────────────────────────────────────────────────

print("Brute-force baseline...")
brute = BruteForceIndex(metric="l2")
brute.add_batch(data)

start = time.perf_counter()
for q in queries:
    brute.query(q, k=K)
brute_time = time.perf_counter() - start
brute_qps = N_QUERIES / brute_time
print(f"  {brute_qps:.0f} QPS\n")


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
ax.set_title(f'Recall vs Throughput — SIFT-10K ({N:,} vectors, {DIM}D, k={K})', fontsize=14)
ax.set_xscale('log')
ax.set_ylim(0.65, 1.02)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('benchmark_sift10k.png', dpi=150, bbox_inches='tight')
print("  Saved: benchmark_sift10k.png")

# Summary
print(f"\n{'='*60}")
print(f"SIFT-10K BENCHMARK RESULTS at ~95% recall:")
print(f"{'='*60}")
for name, points in [("My HNSW", my_points), ("hnswlib", lib_points), ("FAISS", faiss_points)]:
    for qps, recall in points:
        if recall >= 0.95:
            print(f"  {name:<15} → {qps:>8.0f} QPS  (recall={recall:.3f})")
            break
print(f"  {'Brute-force':<15} → {brute_qps:>8.0f} QPS  (recall=1.000)")
print(f"{'='*60}")
