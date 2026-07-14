"""
FastAPI backend for the Mini Vector DB visualization.

Endpoints:
  POST /search       — semantic search, returns results + algorithm events
  POST /add          — add text to the index, returns insert events
  GET  /stats        — index statistics
  GET  /graph        — full graph structure for initial render
  WS   /ws/search    — WebSocket streaming of search events (real-time animation)
  WS   /ws/add       — WebSocket streaming of insert events
"""

import os
import json
import asyncio
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from hnsw_instrumented import InstrumentedHNSW

# ──────────────────────────────────────────────────────────────────────
# Initialize
# ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="Mini Vector DB", version="1.0")

# Allow frontend dev server to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
INDEX_PATH = os.path.join(os.path.dirname(__file__), "saved_index")
POSITIONS_PATH = os.path.join(os.path.dirname(__file__), "positions.npy")
TEXTS_PATH = os.path.join(os.path.dirname(__file__), "texts.json")

embedder = None
index: InstrumentedHNSW = None
texts: list[str] = []          # texts[node_id] = original text
positions: np.ndarray = None   # 2D UMAP positions for visualization


def load_or_create_index():
    """Load saved index or create a fresh one."""
    global index, texts, positions

    if os.path.exists(os.path.join(INDEX_PATH, "metadata.pkl")):
        # Load existing
        index = InstrumentedHNSW.load(INDEX_PATH)
        if os.path.exists(TEXTS_PATH):
            with open(TEXTS_PATH, "r") as f:
                texts = json.load(f)
        if os.path.exists(POSITIONS_PATH):
            positions = np.load(POSITIONS_PATH)
        print(f"Loaded index: {index}, {len(texts)} texts")
    else:
        # Create fresh
        index = InstrumentedHNSW(M=16, ef_construction=200, metric="l2", seed=42)
        texts = []
        positions = np.zeros((0, 2))
        print("Created fresh index")


@app.on_event("startup")
async def startup():
    global embedder
    try:
        from embedder import Embedder
        embedder = Embedder()
        print(f"Embedder ready (dim={embedder.dim})")
    except Exception as e:
        embedder = None
        print(f"Embedder not available (search-only mode): {e}")
    load_or_create_index()


# ──────────────────────────────────────────────────────────────────────
# REST Endpoints
# ──────────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    k: int = 5
    ef_search: int = 50


class AddRequest(BaseModel):
    text: str


@app.post("/search")
async def search(req: SearchRequest):
    """Semantic search — returns results and algorithm events."""
    if embedder is None:
        # Search-only mode: use a random vector from the index as query
        # (demo mode — shows the traversal animation without needing the model)
        import random as rnd
        random_id = rnd.randint(0, len(index) - 1)
        query_vec = index.vectors[random_id]
    else:
        query_vec = embedder.embed(req.query)

    results, events = index.search_with_events(query_vec, k=req.k, ef_search=req.ef_search)

    return {
        "query": req.query,
        "results": [
            {"node_id": nid, "distance": round(dist, 4), "text": texts[nid] if nid < len(texts) else ""}
            for nid, dist in results
        ],
        "events": events,
        "stats": {
            "nodes_visited": len([e for e in events if e["type"] == "node_visited"]),
            "total_nodes": len(index),
        }
    }


@app.post("/add")
async def add_text(req: AddRequest):
    """Add text to the index — returns insert events."""
    global positions

    if embedder is None:
        return {"error": "Embedder not available. Running in search-only mode (demo deployment)."}

    vec = embedder.embed(req.text)
    events = index.insert_with_events(vec)

    node_id = len(texts)
    texts.append(req.text)

    # Compute a simple 2D position for the new node
    # (use first 2 components of the vector as a rough projection)
    # Real UMAP would be recomputed periodically
    if positions is not None and len(positions) > 0:
        new_pos = np.array([[vec[0], vec[1]]])
        positions = np.vstack([positions, new_pos])
    else:
        positions = np.array([[vec[0], vec[1]]])

    # Auto-save
    _save_state()

    return {
        "node_id": node_id,
        "text": req.text,
        "level": index.levels[node_id],
        "events": events,
    }


@app.get("/stats")
async def stats():
    """Index statistics."""
    layer_stats = []
    for layer in range(len(index.graphs)):
        n_nodes = len(index.graphs[layer])
        avg_edges = (
            np.mean([len(f) for f in index.graphs[layer].values()])
            if index.graphs[layer] else 0
        )
        layer_stats.append({"layer": layer, "nodes": n_nodes, "avg_edges": round(avg_edges, 1)})

    return {
        "total_vectors": len(index),
        "dimensions": embedder.dim,
        "max_level": index.max_level,
        "M": index.M,
        "entry_point": index.entry_point,
        "layers": layer_stats,
    }


@app.get("/graph")
async def get_graph():
    """Full graph structure for initial frontend render."""
    nodes = []
    for i in range(len(index.vectors)):
        pos = positions[i].tolist() if positions is not None and i < len(positions) else [0, 0]
        nodes.append({
            "id": i,
            "text": texts[i] if i < len(texts) else "",
            "level": index.levels[i],
            "position": pos,
        })

    edges = []
    for layer in range(len(index.graphs)):
        for node_id, friends in index.graphs[layer].items():
            for friend_id in friends:
                # Avoid duplicates (only add if node_id < friend_id)
                if node_id < friend_id:
                    edges.append({
                        "source": node_id,
                        "target": friend_id,
                        "layer": layer,
                    })

    return {
        "nodes": nodes,
        "edges": edges,
        "entry_point": index.entry_point,
        "max_level": index.max_level,
    }


# ──────────────────────────────────────────────────────────────────────
# WebSocket Endpoints (real-time event streaming)
# ──────────────────────────────────────────────────────────────────────

@app.websocket("/ws/search")
async def ws_search(websocket: WebSocket):
    """Stream search events one-by-one for real-time animation."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            query_text = data.get("query", "")
            k = data.get("k", 5)
            ef_search = data.get("ef_search", 50)

            query_vec = embedder.embed(query_text)
            results, events = index.search_with_events(query_vec, k=k, ef_search=ef_search)

            # Stream events one at a time with a small delay for animation
            for event in events:
                await websocket.send_json(event)
                await asyncio.sleep(0.05)  # 50ms between events

            # Send final results
            await websocket.send_json({
                "type": "final_results",
                "results": [
                    {"node_id": nid, "distance": round(dist, 4),
                     "text": texts[nid] if nid < len(texts) else ""}
                    for nid, dist in results
                ]
            })
    except WebSocketDisconnect:
        pass


@app.websocket("/ws/add")
async def ws_add(websocket: WebSocket):
    """Stream insert events for real-time animation."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            text = data.get("text", "")

            vec = embedder.embed(text)
            events = index.insert_with_events(vec)

            node_id = len(texts)
            texts.append(text)

            # Stream events
            for event in events:
                await websocket.send_json(event)
                await asyncio.sleep(0.05)

            _save_state()

            await websocket.send_json({
                "type": "insert_done",
                "node_id": node_id,
                "text": text,
                "level": index.levels[node_id],
            })
    except WebSocketDisconnect:
        pass


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _save_state():
    """Persist index + texts + positions to disk."""
    index.save(INDEX_PATH)
    with open(TEXTS_PATH, "w") as f:
        json.dump(texts, f)
    if positions is not None and len(positions) > 0:
        np.save(POSITIONS_PATH, positions)


# ──────────────────────────────────────────────────────────────────────
# Serve frontend static files
# ──────────────────────────────────────────────────────────────────────

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.exists(FRONTEND_DIR):
    from fastapi.responses import FileResponse

    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets")), name="assets")


# ──────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)
