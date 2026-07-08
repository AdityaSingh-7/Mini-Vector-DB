"""
Instrumented HNSW — emits step-by-step events for frontend visualization.

Wraps the core HNSWIndex and records each step of search/insert as an event
that can be streamed over WebSocket to animate the algorithm in real-time.
"""

from dataclasses import dataclass, field, asdict
from typing import Literal
import numpy as np
from hnsw import HNSWIndex


@dataclass
class Event:
    """A single step in the HNSW algorithm, serializable to JSON."""
    type: str           # event type name
    data: dict = field(default_factory=dict)

    def to_dict(self):
        return {"type": self.type, **self.data}


class InstrumentedHNSW(HNSWIndex):
    """
    HNSW index that records algorithm events for visualization.

    Use search_with_events() and insert_with_events() to get both
    results AND the event trace.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._events: list[Event] = []
        self._recording = False

    def _emit(self, event_type: str, **data):
        """Record an event if we're currently recording."""
        if self._recording:
            self._events.append(Event(type=event_type, data=data))

    # ──────────────────────────────────────────────────────────────────
    # Override _search_layer to emit per-step events
    # ──────────────────────────────────────────────────────────────────

    def _search_layer(self, query, entry_points, layer, ef):
        """Instrumented version — emits events during search."""
        import heapq

        visited = set(entry_points)

        candidates = []
        for ep in entry_points:
            dist = self._distance_to_query(query, ep)
            heapq.heappush(candidates, (dist, ep))
            self._emit("node_visited", node_id=ep, layer=layer, distance=round(dist, 4))

        results = []
        for ep in entry_points:
            dist = self._distance_to_query(query, ep)
            heapq.heappush(results, (-dist, ep))

        while candidates:
            candidate_dist, candidate_id = heapq.heappop(candidates)

            furthest_result_dist = -results[0][0]
            if candidate_dist > furthest_result_dist:
                break

            self._emit("node_expanded", node_id=candidate_id, layer=layer,
                       distance=round(candidate_dist, 4))

            neighbours = self.graphs[layer].get(candidate_id, [])
            for neighbour_id in neighbours:
                if neighbour_id in visited:
                    continue
                visited.add(neighbour_id)

                neighbour_dist = self._distance_to_query(query, neighbour_id)

                furthest_result_dist = -results[0][0]

                if len(results) < ef or neighbour_dist < furthest_result_dist:
                    heapq.heappush(candidates, (neighbour_dist, neighbour_id))
                    heapq.heappush(results, (-neighbour_dist, neighbour_id))

                    self._emit("node_visited", node_id=neighbour_id, layer=layer,
                               distance=round(neighbour_dist, 4),
                               from_node=candidate_id)

                    if len(results) > ef:
                        heapq.heappop(results)

        final = [(-dist, node_id) for dist, node_id in results]
        final.sort()
        return final

    # ──────────────────────────────────────────────────────────────────
    # Public instrumented methods
    # ──────────────────────────────────────────────────────────────────

    def search_with_events(
        self, query_vector: np.ndarray, k: int = 5, ef_search: int | None = None
    ) -> tuple[list[tuple[int, float]], list[dict]]:
        """
        Search and return both results AND the event trace.

        Returns:
            (results, events) where:
              results = [(node_id, distance), ...] (same as query())
              events = [{"type": "...", ...}, ...] (for animation)
        """
        self._events = []
        self._recording = True

        if self.entry_point is None:
            self._recording = False
            return [], []

        if ef_search is None:
            ef_search = max(k, self.ef_construction)
        ef_search = max(ef_search, k)

        query_vector = np.array(query_vector, dtype=np.float32)
        current_entry = self.entry_point

        # Emit search start
        self._emit("search_start", entry_point=self.entry_point,
                   top_layer=self.max_level)

        # Phase 1: descend
        for layer in range(self.max_level, 0, -1):
            self._emit("layer_start", layer=layer, entry_node=current_entry)
            results = self._search_layer(
                query=query_vector, entry_points=[current_entry], layer=layer, ef=1
            )
            current_entry = results[0][1]
            self._emit("layer_drop", from_layer=layer, to_layer=layer - 1,
                       node=current_entry)

        # Phase 2: final search on layer 0
        self._emit("layer_start", layer=0, entry_node=current_entry)
        results = self._search_layer(
            query=query_vector, entry_points=[current_entry], layer=0, ef=ef_search
        )

        top_k = results[:k]
        result_list = [(node_id, dist) for dist, node_id in top_k]

        self._emit("search_complete",
                   results=[{"node_id": nid, "distance": round(d, 4)} for nid, d in result_list],
                   nodes_visited=len([e for e in self._events if e.type == "node_visited"]))

        self._recording = False
        events = [e.to_dict() for e in self._events]
        self._events = []
        return result_list, events

    def insert_with_events(self, vector: np.ndarray) -> list[dict]:
        """
        Insert a vector and return the event trace.

        Returns:
            List of event dicts for animation.
        """
        self._events = []
        self._recording = True

        new_id = len(self.vectors)
        new_level = self._random_level()

        self._emit("insert_start", node_id=new_id, level=new_level)

        # Do the actual insert (parent class logic, but we're recording)
        self._recording = False
        self.vectors.append(np.array(vector, dtype=np.float32))
        self.levels.append(new_level)

        while len(self.graphs) <= new_level:
            self.graphs.append({})
        for layer in range(new_level + 1):
            self.graphs[layer][new_id] = []

        if self.entry_point is None:
            self.entry_point = new_id
            self.max_level = new_level
            self._recording = True
            self._emit("insert_complete", node_id=new_id, level=new_level, edges={})
            self._recording = False
            events = [e.to_dict() for e in self._events]
            self._events = []
            return events

        self._recording = True
        current_entry = self.entry_point

        # Phase 1: descend
        for layer in range(self.max_level, new_level, -1):
            results = self._search_layer(
                query=vector, entry_points=[current_entry], layer=layer, ef=1
            )
            current_entry = results[0][1]

        # Phase 2: connect
        edges_by_layer = {}
        for layer in range(min(new_level, self.max_level), -1, -1):
            results = self._search_layer(
                query=vector, entry_points=[current_entry], layer=layer,
                ef=self.ef_construction
            )

            M_layer = self.M_max0 if layer == 0 else self.M
            neighbours = self._select_neighbours_heuristic(
                query_vector=vector, candidates=results, M=M_layer
            )

            self.graphs[layer][new_id] = neighbours
            edges_by_layer[layer] = neighbours

            for neighbour_id in neighbours:
                self._emit("edge_added", from_node=new_id, to_node=neighbour_id, layer=layer)

                neighbour_friends = self.graphs[layer][neighbour_id]
                neighbour_friends.append(new_id)

                M_max = self.M_max0 if layer == 0 else self.M
                if len(neighbour_friends) > M_max:
                    neighbour_vector = self.vectors[neighbour_id]
                    friend_candidates = [
                        (self._distance(neighbour_vector, self.vectors[fid]), fid)
                        for fid in neighbour_friends
                    ]
                    friend_candidates.sort()
                    new_friends = self._select_neighbours_heuristic(
                        query_vector=neighbour_vector,
                        candidates=friend_candidates,
                        M=M_max,
                    )
                    self.graphs[layer][neighbour_id] = new_friends

            current_entry = results[0][1]

        if new_level > self.max_level:
            self.entry_point = new_id
            self.max_level = new_level

        self._emit("insert_complete", node_id=new_id, level=new_level,
                   edges={str(k): v for k, v in edges_by_layer.items()})

        self._recording = False
        events = [e.to_dict() for e in self._events]
        self._events = []
        return events
