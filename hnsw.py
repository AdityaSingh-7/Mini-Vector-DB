"""
v1 — HNSW (Hierarchical Navigable Small World) Index

An approximate nearest-neighbour index that builds a layered graph.
Instead of comparing the query to every vector (brute-force),
we "walk" the graph greedily toward the query — visiting far fewer nodes.

Built from scratch following the original paper:
  Malkov & Yashunin, "Efficient and robust approximate nearest neighbor
  using Hierarchical Navigable Small World graphs" (2016/2018).
"""

import math
import random
import numpy as np
from typing import Literal


class HNSWIndex:
    """Hierarchical Navigable Small World graph for approximate nearest-neighbour search."""

    def __init__(
        self,
        M: int = 16,
        ef_construction: int = 200,
        metric: Literal["l2", "cosine"] = "l2",
        seed: int | None = None,
    ):
        """
        Args:
            M: Max edges per node per layer. Controls graph density.
               Higher M → better recall, more memory, slower insert.
               Typical range: 12–48.

            ef_construction: Search width during insertion (how many candidates
               we consider when picking neighbours for a new node).
               Higher → better graph quality, slower build.
               Typical range: 100–500.

            metric: Distance function ("l2" or "cosine").

            seed: Random seed for reproducibility (affects level assignment).
        """
        self.M = M
        self.M_max0 = 2 * M  # Layer 0 gets double the edges (paper recommendation)
        self.ef_construction = ef_construction
        self.metric = metric

        # Level generation scaling factor: 1 / ln(M)
        # This makes P(level >= L) = (1/M)^L
        self._level_mult = 1.0 / math.log(M)

        # Random number generator (separate from global state)
        self._rng = random.Random(seed)

        # --- Storage ---
        self.vectors: list[np.ndarray] = []  # vectors[id] = the raw vector
        self.levels: list[int] = []          # levels[id] = max layer for this node

        # graphs[layer][node_id] = list of neighbour IDs on that layer
        self.graphs: list[dict[int, list[int]]] = []

        # Entry point: the node with the highest layer
        self.entry_point: int | None = None
        self.max_level: int = -1  # highest layer that exists

    def _random_level(self) -> int:
        """
        Pick a random layer for a new node.

        Uses: floor(-ln(uniform(0,1)) * (1/ln(M)))

        This gives an exponential distribution:
          - ~93% of nodes → level 0 only  (with M=16)
          - ~6% → level 1
          - ~0.4% → level 2
          - etc.
        """
        # random() is in (0, 1] — we use 1-random() to avoid ln(0)
        r = self._rng.random()
        # Clamp to avoid ln(0) if r happens to be exactly 0
        r = max(r, 1e-10)
        return int(-math.log(r) * self._level_mult)

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute distance between two vectors."""
        if self.metric == "l2":
            diff = a - b
            return float(np.dot(diff, diff))  # squared L2 (skip sqrt — ranking same)
        elif self.metric == "cosine":
            # cosine distance = 1 - cosine_similarity
            dot = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a < 1e-10 or norm_b < 1e-10:
                return 1.0
            return 1.0 - dot / (norm_a * norm_b)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def _distance_to_query(self, query: np.ndarray, node_id: int) -> float:
        """Distance from query vector to a stored node."""
        return self._distance(query, self.vectors[node_id])

    # ──────────────────────────────────────────────────────────────────────
    # CORE OPERATION: search within a single layer (beam search / greedy walk)
    # ──────────────────────────────────────────────────────────────────────

    def _search_layer(
        self, query: np.ndarray, entry_points: list[int], layer: int, ef: int
    ) -> list[tuple[float, int]]:
        """
        Beam search on one layer of the graph.

        Starting from entry_points, greedily explore neighbours closest to
        the query. Maintains a beam of width `ef` — the ef closest nodes
        seen so far.

        Args:
            query: The vector we're searching for neighbours of.
            entry_points: Node IDs to start the search from.
            layer: Which layer of the graph to search on.
            ef: Beam width — how many candidates to track.

        Returns:
            List of (distance, node_id) for the ef closest nodes found,
            sorted nearest-first.
        """
        import heapq

        # --- Initialise with entry points ---
        # visited: set of node IDs we've already expanded (looked at friends of)
        visited = set(entry_points)

        # candidates: min-heap — "what to explore next" (closest first)
        # Python heapq is a min-heap, so smallest distance pops first. Perfect.
        candidates = []
        for ep in entry_points:
            dist = self._distance_to_query(query, ep)
            heapq.heappush(candidates, (dist, ep))

        # results: max-heap — "best ef nodes found so far" (furthest first for easy eviction)
        # We NEGATE distances so that heapq (min-heap) gives us the largest distance first.
        results = []
        for ep in entry_points:
            dist = self._distance_to_query(query, ep)
            heapq.heappush(results, (-dist, ep))  # negate!

        # --- Main loop: expand closest unvisited candidate ---
        while candidates:
            # Peek at closest candidate
            candidate_dist, candidate_id = heapq.heappop(candidates)

            # STOP CONDITION: if this candidate is farther than our worst result,
            # nothing left can improve our answer.
            # (results is max-heap via negation, so -results[0][0] = furthest result distance)
            furthest_result_dist = -results[0][0]
            if candidate_dist > furthest_result_dist:
                break

            # Expand this candidate: look at all its friends on this layer
            neighbours = self.graphs[layer].get(candidate_id, [])
            for neighbour_id in neighbours:
                if neighbour_id in visited:
                    continue  # already explored this one
                visited.add(neighbour_id)

                neighbour_dist = self._distance_to_query(query, neighbour_id)

                # Should we add this neighbour to our results?
                furthest_result_dist = -results[0][0]

                if len(results) < ef or neighbour_dist < furthest_result_dist:
                    # Yes — it's either better than our worst, or we have room
                    heapq.heappush(candidates, (neighbour_dist, neighbour_id))
                    heapq.heappush(results, (-neighbour_dist, neighbour_id))

                    # If results is over capacity, kick the worst (furthest)
                    if len(results) > ef:
                        heapq.heappop(results)  # removes largest (most negative = furthest)

        # Convert results from max-heap (negated) to sorted list (nearest first)
        final = [(-dist, node_id) for dist, node_id in results]
        final.sort()  # sort by distance ascending (nearest first)
        return final

    # ──────────────────────────────────────────────────────────────────────
    # NEIGHBOUR SELECTION: which candidates to keep as friends
    # ──────────────────────────────────────────────────────────────────────

    def _select_neighbours_simple(
        self, candidates: list[tuple[float, int]], M: int
    ) -> list[int]:
        """
        Simple selection: just take the M closest candidates.

        Args:
            candidates: List of (distance, node_id), sorted nearest-first.
            M: How many neighbours to keep.

        Returns:
            List of node_ids for the chosen neighbours.
        """
        return [node_id for _, node_id in candidates[:M]]

    def _select_neighbours_heuristic(
        self, query_vector: np.ndarray, candidates: list[tuple[float, int]], M: int
    ) -> list[int]:
        """
        Heuristic selection: prefer diversity over pure closeness.

        For each candidate (processed closest-first), keep it ONLY if it's
        closer to the query than to any already-selected neighbour.
        This spreads connections in different directions, preventing clusters
        of redundant edges and reducing local minima.

        If we don't fill M slots with the heuristic alone (too aggressive),
        we fall back to filling remaining slots with the closest rejected ones.

        Args:
            query_vector: The vector being inserted (or the query point).
            candidates: List of (distance, node_id), sorted nearest-first.
            M: How many neighbours to keep.

        Returns:
            List of node_ids for the chosen neighbours.
        """
        selected = []    # nodes we've accepted
        rejected = []    # nodes skipped by heuristic (might use as fallback)

        for dist_to_query, candidate_id in candidates:
            if len(selected) >= M:
                break

            candidate_vector = self.vectors[candidate_id]

            # Check: is this candidate closer to ANY already-selected neighbour
            # than to the query? If yes, it's "covered" — skip it.
            is_covered = False
            for selected_id in selected:
                dist_to_selected = self._distance(candidate_vector, self.vectors[selected_id])
                if dist_to_selected < dist_to_query:
                    # This candidate is closer to an existing selection than to query.
                    # It's redundant — we can reach it through the selected node.
                    is_covered = True
                    break

            if not is_covered:
                selected.append(candidate_id)
            else:
                rejected.append(candidate_id)

        # Fallback: if heuristic was too aggressive and we have < M,
        # fill remaining slots with closest rejected candidates
        for candidate_id in rejected:
            if len(selected) >= M:
                break
            selected.append(candidate_id)

        return selected

    # ──────────────────────────────────────────────────────────────────────
    # INSERT: add a vector to the index and connect it into the graph
    # ──────────────────────────────────────────────────────────────────────

    def insert(self, vector: np.ndarray):
        """
        Insert a vector into the HNSW index.

        Steps:
          1. Assign a random level (coin flip)
          2. Descend from top layer to the new node's level (ef=1, just navigating)
          3. On each layer the new node lives on: search for neighbours,
             select the best M, add bidirectional edges
          4. Update entry point if this node is the new highest

        Args:
            vector: The vector to insert (numpy array).
        """
        import heapq

        # Store the vector and assign an ID
        new_id = len(self.vectors)
        self.vectors.append(np.array(vector, dtype=np.float32))

        # Assign a random level
        new_level = self._random_level()
        self.levels.append(new_level)

        # Ensure we have enough layers in our graph structure
        while len(self.graphs) <= new_level:
            self.graphs.append({})  # add empty layers as needed

        # Register this node on all layers it belongs to (with empty neighbour lists)
        for layer in range(new_level + 1):
            self.graphs[layer][new_id] = []

        # --- Special case: first node ever inserted ---
        if self.entry_point is None:
            self.entry_point = new_id
            self.max_level = new_level
            return  # nothing to connect to

        # --- Phase 1: DESCEND through layers above new_level ---
        # Use ef=1 (pure greedy) — just navigating to the right neighbourhood
        current_entry = self.entry_point

        for layer in range(self.max_level, new_level, -1):
            # Search this layer with ef=1 → get single closest node
            results = self._search_layer(
                query=vector, entry_points=[current_entry], layer=layer, ef=1
            )
            current_entry = results[0][1]  # closest node becomes entry for layer below

        # --- Phase 2: CONNECT on layers where new node lives ---
        # Search with full ef_construction width, then select M neighbours

        for layer in range(min(new_level, self.max_level), -1, -1):
            # Search this layer to find candidates
            results = self._search_layer(
                query=vector,
                entry_points=[current_entry],
                layer=layer,
                ef=self.ef_construction,
            )

            # How many neighbours to keep on this layer?
            M_layer = self.M_max0 if layer == 0 else self.M

            # Select neighbours using the heuristic
            neighbours = self._select_neighbours_heuristic(
                query_vector=vector, candidates=results, M=M_layer
            )

            # Connect new_node → neighbours
            self.graphs[layer][new_id] = neighbours

            # Connect neighbours → new_node (bidirectional)
            for neighbour_id in neighbours:
                neighbour_friends = self.graphs[layer][neighbour_id]
                neighbour_friends.append(new_id)

                # If neighbour now has too many edges, trim it
                M_max = self.M_max0 if layer == 0 else self.M
                if len(neighbour_friends) > M_max:
                    # Re-select using the heuristic from the neighbour's perspective
                    neighbour_vector = self.vectors[neighbour_id]
                    # Build candidate list: (distance_to_neighbour, friend_id)
                    friend_candidates = [
                        (self._distance(neighbour_vector, self.vectors[fid]), fid)
                        for fid in neighbour_friends
                    ]
                    friend_candidates.sort()  # sort by distance

                    # Re-select best M_max
                    new_friends = self._select_neighbours_heuristic(
                        query_vector=neighbour_vector,
                        candidates=friend_candidates,
                        M=M_max,
                    )
                    self.graphs[layer][neighbour_id] = new_friends

            # Use the closest result as entry point for the next layer down
            current_entry = results[0][1]

        # --- Update entry point if new node has the highest level ---
        if new_level > self.max_level:
            self.entry_point = new_id
            self.max_level = new_level

    # ──────────────────────────────────────────────────────────────────────
    # QUERY: find the k nearest neighbours to a query vector
    # ──────────────────────────────────────────────────────────────────────

    def query(
        self, query_vector: np.ndarray, k: int = 5, ef_search: int | None = None
    ) -> list[tuple[int, float]]:
        """
        Find the k approximate nearest neighbours to query_vector.

        Algorithm:
          1. Start at entry point on the highest layer.
          2. Descend layer by layer with ef=1 (greedy — just navigating).
          3. On layer 0, search with ef=ef_search (wide beam — finding results).
          4. Return the top-k from that search.

        Args:
            query_vector: The vector to search for.
            k: Number of nearest neighbours to return.
            ef_search: Beam width for the final search. Higher = better recall,
                       slower query. Defaults to max(k, ef_construction).
                       Must be >= k (you can't return k results from a beam
                       narrower than k).

        Returns:
            List of (index, distance) tuples, sorted nearest-first.
            Same format as BruteForceIndex.query() for easy comparison.
        """
        if self.entry_point is None:
            return []  # empty index

        # ef_search must be at least k (need enough candidates to pick k from)
        if ef_search is None:
            ef_search = max(k, self.ef_construction)
        ef_search = max(ef_search, k)

        query_vector = np.array(query_vector, dtype=np.float32)

        # --- Phase 1: DESCEND from top layer to layer 1 (ef=1, just navigating) ---
        current_entry = self.entry_point

        for layer in range(self.max_level, 0, -1):
            results = self._search_layer(
                query=query_vector,
                entry_points=[current_entry],
                layer=layer,
                ef=1,
            )
            current_entry = results[0][1]  # closest node → entry for next layer

        # --- Phase 2: SEARCH layer 0 with full ef_search width ---
        results = self._search_layer(
            query=query_vector,
            entry_points=[current_entry],
            layer=0,
            ef=ef_search,
        )

        # Return top-k, formatted as (index, distance) to match BruteForceIndex
        top_k = results[:k]
        return [(node_id, dist) for dist, node_id in top_k]

    # ──────────────────────────────────────────────────────────────────────
    # PERSISTENCE: save/load index to disk
    # ──────────────────────────────────────────────────────────────────────

    def save(self, path: str):
        """
        Save the entire index to disk.

        Stores:
          - vectors as a numpy .npy file
          - graph structure + metadata as a pickle file

        Args:
            path: Directory path to save into (created if doesn't exist).
        """
        import os
        import pickle

        os.makedirs(path, exist_ok=True)

        # Save vectors as a numpy array (fast, compact)
        vectors_array = np.vstack(self.vectors) if self.vectors else np.array([])
        np.save(os.path.join(path, "vectors.npy"), vectors_array)

        # Save everything else as pickle
        metadata = {
            "M": self.M,
            "M_max0": self.M_max0,
            "ef_construction": self.ef_construction,
            "metric": self.metric,
            "level_mult": self._level_mult,
            "levels": self.levels,
            "graphs": self.graphs,
            "entry_point": self.entry_point,
            "max_level": self.max_level,
        }
        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)

    @classmethod
    def load(cls, path: str) -> "HNSWIndex":
        """
        Load a saved index from disk.

        Args:
            path: Directory path where the index was saved.

        Returns:
            A fully reconstructed HNSWIndex ready for queries.
        """
        import os
        import pickle

        # Load vectors
        vectors_array = np.load(os.path.join(path, "vectors.npy"))

        # Load metadata
        with open(os.path.join(path, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)

        # Reconstruct the index
        index = cls(
            M=metadata["M"],
            ef_construction=metadata["ef_construction"],
            metric=metadata["metric"],
        )
        index.M_max0 = metadata["M_max0"]
        index._level_mult = metadata["level_mult"]
        index.levels = metadata["levels"]
        index.graphs = metadata["graphs"]
        index.entry_point = metadata["entry_point"]
        index.max_level = metadata["max_level"]

        # Rebuild vectors list from the numpy array
        if vectors_array.size > 0:
            index.vectors = [vectors_array[i] for i in range(len(vectors_array))]
        else:
            index.vectors = []

        return index

    def __len__(self):
        return len(self.vectors)

    def __repr__(self):
        return (
            f"HNSWIndex(M={self.M}, ef_construction={self.ef_construction}, "
            f"metric='{self.metric}', vectors={len(self)}, "
            f"layers={self.max_level + 1})"
        )
