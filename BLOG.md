# How HNSW Works: Building a Vector Search Engine from Scratch

*I implemented the algorithm used by Pinecone, Weaviate, and pgvector's HNSW index — from scratch in Python. Here's what I learned.*

---

## The Problem

You have 1 million vectors (lists of numbers representing text, images, or anything). A query arrives. Find the 10 most similar. Fast.

**Brute force** compares the query to every single vector — 1 million distance calculations per query. On my machine, brute force on 10K vectors does ~2,000 QPS — but that scales linearly, so at 1M vectors you'd be looking at ~20 QPS. Unusable for real-time applications.

**The solution:** build a graph and walk it. Visit 100–300 nodes instead of 1,000,000. That's what HNSW (Hierarchical Navigable Small World) does.

Along the way I learned why production systems reach for C++, SIMD vectorization, and product quantization to close the gap between "correct algorithm" and "fast system" — more on that below.

---

## The Core Idea in 30 Seconds

Imagine finding a specific café in Paris, starting from New York:

1. **Fly** (Layer 2): New York → Paris. One big hop — you're in the right country.
2. **Metro** (Layer 1): Paris → Le Marais. A few medium hops — right neighbourhood.
3. **Walk** (Layer 0): Le Marais → the café. Short steps — precise answer.

HNSW does exactly this with vectors. It builds a layered graph where:
- **Top layers** have few nodes with long-range connections (highways)
- **Bottom layer** has all nodes with short-range connections (local streets)
- **Search** starts at the top and descends layer by layer, getting more precise

---

## How the Graph is Built

### Each vector gets a random level

When you insert a vector, a coin flip decides how high it reaches:

```
level = floor(-ln(random()) × 1/ln(M))
```

This creates an exponential pyramid (for M=16):
- ~94% of nodes only reach Layer 0
- ~6% reach Layer 1 or higher
- ~0.4% reach Layer 2 or higher

The formula is elegant — one line creates the entire sparse-top/dense-bottom hierarchy automatically. No tuning needed.

**Why exponential?** Because the top layer needs very few nodes (just enough to cross the space in a few hops), and the bottom needs everyone (for precise local search). The exponential distribution is the natural way to get this pyramid shape.

### Each vector connects to its M nearest neighbours

On each layer it lives on, a new vector finds its closest nodes and connects to them. But "closest" isn't the whole story — HNSW uses a **diversity heuristic**:

> "Is this candidate closer to me than to any neighbour I've already selected?"
> If yes → keep it (it's in a new direction).
> If no → skip it (I can already reach it through someone I've picked).

This prevents all edges from clustering in one direction, which would create dead ends for greedy search.

---

## How Search Works

Given a query vector, find the k nearest neighbours:

**Phase 1 — Descend** (ef=1 per layer, just navigating):
```
Layer 2: Start at entry point → greedy hop (2-3 hops) → closest node
                ↓ drop
Layer 1: Continue → greedy hop (5-10 hops) → closest node  
                ↓ drop
Layer 0: Now we're in the right neighbourhood
```

**Phase 2 — Search** (ef=ef_search, wide beam on Layer 0):
```
Layer 0: Beam search with width ef_search
         Explore friends of friends, keep the best ef candidates
         Return top-k when nothing unexplored can improve results
```

Total nodes visited: ~100–300 out of 1,000,000. That's the win.

### The ef parameter: one dial controls everything

The same graph, the same algorithm — just a wider or narrower beam:

| ef_search | Nodes visited | Recall@10 | Speed |
|-----------|--------------|-----------|-------|
| 10 | ~40 | 95.6% | Fastest |
| 50 | ~110 | 99.9% | Medium |
| 75 | ~142 | 100% | Slower |
| 200 | ~306 | 100% | Slowest |

You literally turn one knob to trade accuracy for speed. No rebuild needed.

---

## My Implementation: What I Built

The full HNSW algorithm from the [original paper](https://arxiv.org/abs/1603.09320) in Python:

- **Brute-force baseline** — exact search for ground-truth comparison
- **HNSW index** — insert, query, persistence (save/load)
- **Heuristic neighbour selection** — diversity-aware edge pruning
- **Benchmark harness** — recall@k vs QPS comparison against hnswlib and FAISS
- **Animated visualization** — React frontend showing the algorithm traverse the graph in real-time
- **Semantic search demo** — 480 embedded texts, searchable via natural language

---

## Benchmark Results

Tested on 10,000 structured vectors (128D, 50 clusters — mimicking real-world data distribution):

| Method | Recall@10 | QPS | Language |
|--------|-----------|-----|----------|
| My HNSW | 95.6% | 6,087 | Pure Python |
| hnswlib | 97.1% | 417,609 | C++ + SIMD |
| FAISS HNSW | 97.3% | 493,675 | C++ + SIMD |
| Brute-force | 100% | 1,967 | Python + NumPy |

### The honest read

My implementation is **~70× slower** than hnswlib/FAISS on throughput. The recall curves match — confirming the algorithm is correct — but the speed gap is real and expected.

**Why 70×?** Three concrete reasons:

1. **Per-node overhead:** Python dict lookups and heap operations vs. C++ flat arrays with cache-friendly memory layout
2. **No SIMD:** Computing L2 distance on 128 floats takes 128 multiplies + 128 adds in Python. With AVX-512, hnswlib does 16 at once — 8× speedup on the tightest inner loop alone.
3. **Interpreter overhead:** Every Python bytecode instruction has ~100ns of dispatch overhead. In a loop that runs millions of times (distance computations × nodes visited), this compounds.

The algorithm is identical. The constant factors aren't. That's the whole story.

### Why my HNSW is faster than brute-force (3× at 95.6% recall)

Even in pure Python with all that overhead, HNSW still beats brute-force. Brute-force does 10,000 distance computations per query (vectorized with NumPy, fast). HNSW does ~40 distance computations with Python overhead per computation. The per-computation cost is higher, but doing 40 instead of 10,000 more than compensates — and this ratio should only improve at larger scale.

At 1M vectors: brute-force would do 1,000,000 computations per query. HNSW should still visit ~100–300 nodes (logarithmic scaling). I haven't benchmarked 1M in my Python implementation yet (build time would be hours), but the hnswlib/FAISS numbers on SIFT-1M confirm the scaling holds.

---

## Key Insights (What I Learned Building This)

### 1. The level formula is the most elegant line in the paper

`floor(-ln(random()) × 1/ln(M))` — one line creates the entire hierarchical structure. No explicit pyramid building, no layer sizing, no rebalancing. The exponential distribution does all the work. I spent an hour understanding why, and it's the thing that made the whole algorithm click.

### 2. The diversity heuristic is critical in high dimensions

Without it, all edges cluster in one direction (toward the densest part of the space), and greedy search gets trapped in local minima. The heuristic forces connections to spread — "I already have a friend in that direction, give me one in a new direction." The graph goes from ~80% recall to ~99% recall just from this change.

### 3. ef is the core insight of HNSW

Same graph. Same algorithm. One parameter: beam width. Turn it up → better recall, slower. Turn it down → worse recall, faster. This is why HNSW is practical — you don't rebuild the index to change the accuracy/speed tradeoff. You just change one number at query time.

### 4. 100% recall on small datasets is expected, not impressive

On 10K vectors with M=16 and ef=75, I get 100% recall — visiting only 142 nodes (1.4% of the dataset). That sounds amazing but it's expected: the graph is well-connected relative to its size (16 edges per node out of 10K = high connectivity ratio), and my clustered data has tight neighbourhoods where the true top-10 are all mutually connected. At 1M vectors, M=16 connects to 0.0016% of the data — the connectivity ratio drops by 100× and reaching 100% recall would require a much larger ef. This is why standard ANN benchmarks use 1M+ vectors, and why I plan to validate on SIFT-1M next.

### 5. The language doesn't matter for correctness, but it dominates throughput

My recall curve matches FAISS exactly. If you only looked at the Y-axis of the chart (recall), you couldn't tell which implementation is Python and which is C++. But look at the X-axis (QPS) and the 70× gap is obvious. This taught me that for systems software, the algorithm is necessary but not sufficient — low-level implementation details (memory layout, SIMD, cache lines) are where real performance lives.

---

## What I'd Change for Production

If this needed to serve real traffic:

1. **Rewrite in C++/Rust** — close the 70× gap with compiled code + SIMD
2. **Memory-mapped storage** — let the OS page the graph in/out for datasets larger than RAM
3. **Product quantization** — compress 128D vectors from 512 bytes to ~64 bytes (8× memory reduction)
4. **Concurrent access** — read-write locks for simultaneous queries + single-writer inserts
5. **Delete support** — tombstone marking + periodic graph compaction

---

## Try It Yourself

The full implementation with animated visualization is on [GitHub](https://github.com/AdityaSingh-7/mini-vecdb).

```bash
git clone https://github.com/AdityaSingh-7/mini-vecdb
cd mini-vecdb && pip install -r requirements.txt
python prepare_demo.py  # embed 480 texts, build the graph
python server.py        # open http://localhost:8080
```

Type a query, watch the algorithm traverse the graph layer by layer, see the results appear.

---

*Built as a learning project to understand what's inside vector databases like Pinecone and Weaviate. The goal was depth of understanding, not production throughput.*
