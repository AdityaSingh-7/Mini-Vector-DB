"""
Benchmark on SIFT-1M (1 million vectors, 128D) — the standard large-scale ANN benchmark.

WARNING: Building our Python HNSW on 1M vectors will take a LONG time (hours).
This script benchmarks hnswlib and FAISS quickly, and optionally benchmarks
our HNSW on a smaller subset or skips it.

Steps:
1. Download sift.tar.gz from: http://corpus-texmex.irisa.fr/sift.tar.gz
   (Alternative: ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz)
   Size: ~168 MB

2. Extract it:
   tar -xzf sift.tar.gz

   You'll get a folder 'sift/' containing:
     sift_base.fvecs         — 1,000,000 vectors (128D)
     sift_query.fvecs        — 10,000 queries
     sift_groundtruth.ivecs  — ground truth (100 nearest neighbours per query)
     sift_learn.fvecs        — training set (not needed)

3. Place the 'sift/' folder inside ~/mini-vecdb/

4. Run:
   cd ~/mini-vecdb
   source .venv/bin/activate
   python benchmark_sift1m.py
"""

import os
import time
import struct
import numpy as np
import matplotlib.pyplot as plt

from hnsw import HNSWIndex
import hnswlib
import faiss


# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────

# Set to True if you want to benchmark your Python HNSW too (SLOW — hours)
# Set to False to only benchmark hnswlib and FAISS (fast — minutes)
BENCHMARK_MY_HNSW = False

# Number of queries to use (sift has 10K, but 500 is enough for stable stats)
MAX_QUERIES = 500

K = 10
EF_VALUES = [10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500]


# ──────────────────────────────────────────────────────────────────────
# File readers
# ──────────────────────────────────────────────────────────────────────

def read_fvecs(filename, max_vectors=None):
    """Read .fvecs file → numpy array of float32."""
    with open(filename, 'rb') as f:
        # Read first vector to get dimensionality
        dim = struct.unpack('i', f.read(4))[0]
        vec_size = 4 + dim * 4  # 4 bytes for dim + dim*4 bytes for floats
        f.seek(0, 2)
        file_size = f.tell()
        n_vectors = file_size // vec_size
        if max_vectors:
            n_vectors = min(n_vectors, max_vectors)
        f.seek(0)

        data = np.zeros((n_vectors, dim), dtype=np.float32)
        for i in range(n_vectors):
            f.read(4)  # skip dim field
            data[i] = np.frombuffer(f.read(dim * 4), dtype=np.float32)
            if (i + 1) % 100000 == 0:
                print(f"    loaded {i+1:,} / {n_vectors:,} vectors...")
    return data


def read_ivecs(filename, max_vectors=None):
    """Read .ivecs file → numpy array of int32."""
    with open(filename, 'rb') as f:
        dim = struct.unpack('i', f.read(4))[0]
        vec_size = 4 + dim * 4
        f.seek(0, 2)
        file_size = f.tell()
        n_vectors = file_size // vec_size
        if max_vectors:
            n_vectors = min(n_vectors, max_vectors)
        f.seek(0)

        data = np.zeros((n_vectors, dim), dtype=np.int32)
        for i in range(n_vectors):
            f.read(4)
            data[i] = np.frombuffer(f.read(dim * 4), dtype=np.int32)
    return data


# ──────────────────────────────────────────────────────────────────────
# Load SIFT-1M
# ──────────────────────────────────────────────────────────────────────

SIFT_DIR = os.path.join(os.path.dirname(__file__), "sift")

if not os.path.exists(SIFT_DIR):
    print("ERROR: 'sift/' folder not found!")
    print()
    print("Download from: http://corpus-texmex.irisa.fr/sift.tar.gz (~168 MB)")
    print("Then:          tar -xzf sift.tar.gz")
    print("Place the 'sift/' folder in this directory.")
    exit(1)

print("Loading SIFT-1M dataset...")
print("  Loading base vectors (1M × 128D)...")
data = read_fvecs(os.path.join(SIFT_DIR, "sift_base.fvecs"))
print(f"  Loading queries...")
queries = read_fvecs(os.path.join(SIFT_DIR, "sift_query.fvecs"), max_vectors=MAX_QUERIES)
print(f"  Loading ground truth...")
gt = read_ivecs(os.path.join(SIFT_DIR, "sift_groundtruth.ivecs"), max_vectors=MAX_QUERIES)

N, DIM = data.shape
N_QUERIES = len(queries)

print(f"\n  Base vectors: {N:,} × {DIM}D")
print(f"  Queries: {N_QUERIES}")
print(f"  k={K}\n")

ground_truth = [set(gt[i, :K]) for i in range(N_QUERIES)]


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
# My HNSW (optional — very slow at 1M)
# ──────────────────────────────────────────────────────────────────────

my_points = []
if BENCHMARK_MY_HNSW:
    print("\n⚠️  Building My HNSW on 1M vectors — this will take HOURS...")
    print("    (Set BENCHMARK_MY_HNSW = False to skip)\n")
    t0 = time.perf_counter()
    my_hnsw = HNSWIndex(M=16, ef_construction=200, metric="l2", seed=42)
    for i, v in enumerate(data):
        my_hnsw.insert(v)
        if (i + 1) % 10000 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            eta = (N - i - 1) / rate
            print(f"    {i+1:,}/{N:,} inserted ({rate:.0f}/s, ETA: {eta/60:.0f} min)")
    build_time = time.perf_counter() - t0
    print(f"  Build: {build_time:.1f}s ({N/build_time:.0f} inserts/sec)")

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
# Plot
# ──────────────────────────────────────────────────────────────────────

print("\nGenerating chart...")
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

if my_points:
    ax.plot([p[0] for p in my_points], [p[1] for p in my_points],
            'o-', color='#e74c3c', linewidth=2, markersize=6, label='My HNSW (Python)')

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
print(f"SIFT-1M BENCHMARK RESULTS at ~95% recall:")
print(f"{'='*60}")
all_methods = [("hnswlib", lib_points), ("FAISS", faiss_points)]
if my_points:
    all_methods = [("My HNSW", my_points)] + all_methods
for name, points in all_methods:
    for qps, recall in points:
        if recall >= 0.95:
            print(f"  {name:<15} → {qps:>8.0f} QPS  (recall={recall:.3f})")
            break
print(f"{'='*60}")
print(f"\nNote: My HNSW {'was' if my_points else 'was NOT'} benchmarked on 1M vectors.")
if not my_points:
    print("  Set BENCHMARK_MY_HNSW = True to include it (takes hours).")
    print("  The 10K benchmark (benchmark.py / benchmark_sift.py) shows the")
    print("  algorithm achieves the same recall curve — the gap is purely")
    print("  Python vs C++ throughput, not algorithmic quality.")
