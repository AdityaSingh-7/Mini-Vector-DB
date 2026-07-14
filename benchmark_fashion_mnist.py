"""
Benchmark on Fashion-MNIST (60K vectors, 784D) — a standard ML benchmark.

This dataset is smaller than SIFT-1M and easy to download.
If you can't get SIFT, this is a solid alternative recognized benchmark.

Download happens automatically via the script (from Amazon S3).
"""

import os
import time
import gzip
import struct
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

from brute_force import BruteForceIndex
from hnsw import HNSWIndex
import hnswlib
import faiss


# ──────────────────────────────────────────────────────────────────────
# Download and load Fashion-MNIST
# ──────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "fashion_mnist")
os.makedirs(DATA_DIR, exist_ok=True)

URLS = {
    "train": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
    "test": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
}


def download_if_needed(name, url):
    path = os.path.join(DATA_DIR, f"{name}.gz")
    if not os.path.exists(path):
        print(f"  Downloading {name}...")
        urllib.request.urlretrieve(url, path)
    return path


def load_mnist_images(path):
    """Load MNIST-format images from gzipped file."""
    with gzip.open(path, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        n_images = struct.unpack('>I', f.read(4))[0]
        n_rows = struct.unpack('>I', f.read(4))[0]
        n_cols = struct.unpack('>I', f.read(4))[0]
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(n_images, n_rows * n_cols).astype(np.float32)
    return data


print("Loading Fashion-MNIST dataset...")
train_path = download_if_needed("train", URLS["train"])
test_path = download_if_needed("test", URLS["test"])

data = load_mnist_images(train_path)      # 60,000 × 784
queries = load_mnist_images(test_path)    # 10,000 × 784

# Use subset for reasonable timing
N_DATA = 10000      # Use first 10K for indexing (comparable to SIFT-10K)
N_QUERIES = 200     # 200 queries for stable recall estimate
K = 10

data = data[:N_DATA]
queries = queries[:N_QUERIES]
DIM = data.shape[1]

print(f"  Data: {data.shape}")
print(f"  Queries: {queries.shape}")
print(f"  k={K}\n")

EF_VALUES = [10, 20, 30, 50, 75, 100, 150, 200, 300, 400]


# ──────────────────────────────────────────────────────────────────────
# Ground truth (brute force)
# ──────────────────────────────────────────────────────────────────────

print("Computing ground truth...")
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
print(f"  Build: {build_time:.1f}s ({N_DATA/build_time:.0f} inserts/sec)")

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
hnsw_lib.init_index(max_elements=N_DATA, M=16, ef_construction=200, random_seed=42)
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
ax.set_title(f'Recall vs Throughput — Fashion-MNIST ({N_DATA:,} vectors, {DIM}D, k={K})', fontsize=14)
ax.set_xscale('log')
ax.set_ylim(0.65, 1.02)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('benchmark_fashion_mnist.png', dpi=150, bbox_inches='tight')
print("  Saved: benchmark_fashion_mnist.png")

print(f"\n{'='*60}")
print(f"FASHION-MNIST BENCHMARK at ~95% recall:")
print(f"{'='*60}")
for name, points in [("My HNSW", my_points), ("hnswlib", lib_points), ("FAISS", faiss_points)]:
    for qps, recall in points:
        if recall >= 0.95:
            print(f"  {name:<15} → {qps:>8.0f} QPS  (recall={recall:.3f})")
            break
print(f"  {'Brute-force':<15} → {brute_qps:>8.0f} QPS  (recall=1.000)")
print(f"{'='*60}")
