import { useState, useEffect, useRef, useCallback } from 'react'
import GraphCanvas from './GraphCanvas'

interface GraphNode {
  id: number
  text: string
  level: number
  position: [number, number]
}

interface GraphEdge {
  source: number
  target: number
  layer: number
}

interface SearchResult {
  node_id: number
  distance: number
  text: string
}

interface AnimEvent {
  type: string
  [key: string]: any
}

// In production (deployed): API is served from the same origin → use ''
// In development (local): API is at localhost:8080 → use 'http://localhost:8080'
// To switch: change this line, then run `npx vite build` in frontend/
const API = ''

function App() {
  const [nodes, setNodes] = useState<GraphNode[]>([])
  const [edges, setEdges] = useState<GraphEdge[]>([])
  const [maxLevel, setMaxLevel] = useState(0)
  const [entryPoint, setEntryPoint] = useState<number | null>(null)

  // Search state
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [isSearching, setIsSearching] = useState(false)

  // Add state
  const [addText, setAddText] = useState('')
  const [isAdding, setIsAdding] = useState(false)

  // Animation state
  const [visitedNodes, setVisitedNodes] = useState<Set<number>>(new Set())
  const [expandedNodes, setExpandedNodes] = useState<Set<number>>(new Set())
  const [resultNodes, setResultNodes] = useState<Set<number>>(new Set())
  const [activeLayer, setActiveLayer] = useState<number | null>(null)
  const [searchPath, setSearchPath] = useState<number[]>([])
  const [nodesVisited, setNodesVisited] = useState(0)

  // Speed control
  const [animSpeed, setAnimSpeed] = useState(50) // ms between events

  // Load initial graph
  useEffect(() => {
    fetch(`${API}/graph`)
      .then(res => res.json())
      .then(data => {
        setNodes(data.nodes)
        setEdges(data.edges)
        setMaxLevel(data.max_level)
        setEntryPoint(data.entry_point)
      })
      .catch(err => console.error('Failed to load graph:', err))
  }, [])

  const clearAnimation = useCallback(() => {
    setVisitedNodes(new Set())
    setExpandedNodes(new Set())
    setResultNodes(new Set())
    setSearchPath([])
    setActiveLayer(null)
    setNodesVisited(0)
  }, [])

  const animateEvents = useCallback(async (events: AnimEvent[]) => {
    clearAnimation()
    const visited = new Set<number>()
    const expanded = new Set<number>()
    const path: number[] = []

    for (const event of events) {
      await new Promise(resolve => setTimeout(resolve, animSpeed))

      switch (event.type) {
        case 'layer_start':
          setActiveLayer(event.layer)
          break

        case 'node_visited':
          visited.add(event.node_id)
          setVisitedNodes(new Set(visited))
          setNodesVisited(visited.size)
          if (event.from_node !== undefined) {
            path.push(event.from_node, event.node_id)
            setSearchPath([...path])
          }
          break

        case 'node_expanded':
          expanded.add(event.node_id)
          setExpandedNodes(new Set(expanded))
          break

        case 'layer_drop':
          setActiveLayer(event.to_layer)
          break

        case 'search_complete':
          const resultIds = new Set<number>(event.results.map((r: any) => r.node_id))
          setResultNodes(resultIds)
          break

        case 'edge_added':
          // For insert animation — refresh edges
          break
      }
    }
  }, [animSpeed, clearAnimation])

  const handleSearch = async () => {
    if (!searchQuery.trim() || isSearching) return
    setIsSearching(true)
    setSearchResults([])
    clearAnimation()

    try {
      const res = await fetch(`${API}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: searchQuery, k: 5, ef_search: 50 }),
      })
      const data = await res.json()

      // Animate the search
      await animateEvents(data.events)
      setSearchResults(data.results)
    } catch (err) {
      console.error('Search failed:', err)
    } finally {
      setIsSearching(false)
    }
  }

  const handleAdd = async () => {
    if (!addText.trim() || isAdding) return
    setIsAdding(true)
    clearAnimation()

    try {
      const res = await fetch(`${API}/add`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: addText }),
      })
      const data = await res.json()

      // Add new node to local state
      const newNode: GraphNode = {
        id: data.node_id,
        text: addText,
        level: data.level,
        position: [Math.random(), Math.random()], // temporary position
      }
      setNodes(prev => [...prev, newNode])

      // Animate insertion
      await animateEvents(data.events)

      // Reload graph to get proper edges
      const graphRes = await fetch(`${API}/graph`)
      const graphData = await graphRes.json()
      setNodes(graphData.nodes)
      setEdges(graphData.edges)
      setMaxLevel(graphData.max_level)

      setAddText('')
    } catch (err) {
      console.error('Add failed:', err)
    } finally {
      setIsAdding(false)
    }
  }

  return (
    <div className="app">
      {/* Header */}
      <div className="header">
        <h1>Mini Vector DB</h1>
        <span className="subtitle">HNSW Visualizer</span>
        <div className="stats">
          <span>{nodes.length} vectors</span>
          <span>{maxLevel + 1} layers</span>
          <span>M=16</span>
          {nodesVisited > 0 && (
            <span className="active-stat">
              {nodesVisited} / {nodes.length} visited
            </span>
          )}
        </div>
      </div>

      {/* Graph Canvas */}
      <div className="graph-panel">
        <GraphCanvas
          nodes={nodes}
          edges={edges}
          maxLevel={maxLevel}
          visitedNodes={visitedNodes}
          expandedNodes={expandedNodes}
          resultNodes={resultNodes}
          activeLayer={activeLayer}
          searchPath={searchPath}
        />
        {/* Layer labels */}
        {/* Legend */}
        <div className="legend">
          <div className="legend-item">
            <div className="legend-dot" style={{ background: 'var(--node-default)', width: '6px', height: '6px' }} />
            <span>Layer 0</span>
          </div>
          <div className="legend-item">
            <div className="legend-dot" style={{ background: '#6366f1', width: '9px', height: '9px' }} />
            <span>Layer 1</span>
          </div>
          <div className="legend-item">
            <div className="legend-dot" style={{ background: 'var(--accent-purple)', width: '12px', height: '12px' }} />
            <span>Layer 2</span>
          </div>
          <div className="legend-item">
            <div className="legend-dot" style={{ background: 'var(--accent-blue)' }} />
            <span>Visited</span>
          </div>
          <div className="legend-item">
            <div className="legend-dot" style={{ background: 'var(--accent-green)' }} />
            <span>Result</span>
          </div>
          <div className="legend-item">
            <div className="legend-dot" style={{ background: 'var(--accent-yellow)', width: '20px', height: '3px', borderRadius: '2px' }} />
            <span>Path</span>
          </div>
        </div>
      </div>

      {/* Side Panel */}
      <div className="side-panel">
        {/* Search */}
        <div className="panel-section">
          <h3>Search</h3>
          <div className="input-group">
            <input
              type="text"
              placeholder="How do black holes form?"
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleSearch()}
              disabled={isSearching}
            />
            <button className="btn" onClick={handleSearch} disabled={isSearching}>
              {isSearching ? '...' : 'Search'}
            </button>
          </div>
        </div>

        {/* Animation Status */}
        {(isSearching || isAdding) && (
          <div className="panel-section">
            <div className="anim-status">
              <div className="pulse-dot" />
              <span>
                {isSearching ? 'Traversing graph' : 'Inserting node'}
                {activeLayer !== null && ` — Layer ${activeLayer}`}
              </span>
            </div>
          </div>
        )}

        {/* Add Text */}
        <div className="panel-section">
          <h3>Add to Index</h3>
          <div className="input-group" style={{ flexDirection: 'column' }}>
            <textarea
              placeholder="Paste any text to add it to the vector index..."
              value={addText}
              onChange={e => setAddText(e.target.value)}
              disabled={isAdding}
            />
            <button
              className="btn btn-add"
              onClick={handleAdd}
              disabled={isAdding || !addText.trim()}
              style={{ alignSelf: 'flex-end' }}
            >
              {isAdding ? 'Embedding...' : 'Add'}
            </button>
          </div>
        </div>

        {/* Speed Control */}
        <div className="panel-section">
          <h3>Animation Speed</h3>
          <div className="speed-control">
            <span>Fast</span>
            <input
              type="range"
              min="10"
              max="200"
              value={animSpeed}
              onChange={e => setAnimSpeed(Number(e.target.value))}
            />
            <span>Slow</span>
          </div>
        </div>

        {/* Results */}
        {searchResults.length > 0 && (
          <div className="panel-section">
            <h3>Results</h3>
            <div className="results-list">
              {searchResults.map((r, i) => (
                <div key={r.node_id} className="result-item">
                  <div className="rank">#{i + 1}</div>
                  <div>{r.text}</div>
                  <div className="distance">Distance: {r.distance.toFixed(4)}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
