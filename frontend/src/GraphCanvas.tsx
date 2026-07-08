import { useMemo } from 'react'
import React from 'react'

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

interface Props {
  nodes: GraphNode[]
  edges: GraphEdge[]
  maxLevel: number
  visitedNodes: Set<number>
  expandedNodes: Set<number>
  resultNodes: Set<number>
  activeLayer: number | null
  searchPath: number[]
}

function GraphCanvas({
  nodes,
  edges,
  maxLevel,
  visitedNodes,
  expandedNodes,
  resultNodes,
}: Props) {
  const isSearching = visitedNodes.size > 0 || resultNodes.size > 0

  // Layout:
  // Top section (y: 30-180): Layer 2 nodes + their edges
  // Middle section (y: 200-350): Layer 1 nodes + their edges
  // Bottom section (y: 380-630): Layer 0 only nodes (the cloud)
  // Nodes with level>0 appear ONLY in their highest layer section

  const nodePositions = useMemo(() => {
    const positions = new Map<number, { x: number; y: number }>()
    if (!nodes.length) return positions

    const width = 1000
    const paddingX = 80

    // Separate nodes by their level
    const layer0Nodes = nodes.filter(n => n.level === 0)
    const layer1Nodes = nodes.filter(n => n.level === 1)
    const layer2Nodes = nodes.filter(n => n.level >= 2)

    // Layer 2: top, spread horizontally
    layer2Nodes.forEach(node => {
      const x = paddingX + node.position[0] * (width - paddingX * 2)
      const y = 60 + node.position[1] * 100
      positions.set(node.id, { x, y })
    })

    // Layer 1: middle band, spread out
    layer1Nodes.forEach(node => {
      const x = paddingX + node.position[0] * (width - paddingX * 2)
      const y = 220 + node.position[1] * 120
      positions.set(node.id, { x, y })
    })

    // Layer 0: bottom cloud, most space
    layer0Nodes.forEach(node => {
      const x = paddingX + node.position[0] * (width - paddingX * 2)
      const y = 400 + node.position[1] * 220
      positions.set(node.id, { x, y })
    })

    return positions
  }, [nodes])

  // Edges: show layer 1+ edges always (few of them), layer 0 edges only during search
  const upperEdges = useMemo(() => edges.filter(e => e.layer > 0), [edges])
  const layer0Edges = useMemo(() => edges.filter(e => e.layer === 0), [edges])

  // Vertical connectors: from layer 2 → layer 1 nodes, and layer 1 → layer 0 area
  // But since layer 1+ nodes don't appear in layer 0, we draw a dotted drop line
  // from a high-layer node down to where it "connects" into layer 0
  const dropLines = useMemo(() => {
    const lines: { id: number; x: number; fromY: number; toY: number }[] = []
    nodes.filter(n => n.level >= 1).forEach(node => {
      const pos = nodePositions.get(node.id)
      if (!pos) return
      // Drop to the top of layer 0 area
      lines.push({ id: node.id, x: pos.x, fromY: pos.y, toY: 390 })
    })
    return lines
  }, [nodes, nodePositions])

  return (
    <svg viewBox="0 0 1000 650" preserveAspectRatio="xMidYMid meet">
      <rect width="1000" height="650" fill="var(--bg-dark)" />

      {/* Layer separator lines */}
      <line x1="50" y1="190" x2="950" y2="190" stroke="var(--border)" strokeWidth="0.5" strokeDasharray="4 8" opacity="0.4" />
      <line x1="50" y1="370" x2="950" y2="370" stroke="var(--border)" strokeWidth="0.5" strokeDasharray="4 8" opacity="0.4" />

      {/* Layer labels */}
      <text x="16" y="105" fontSize="10" fill="var(--text-secondary)" opacity="0.5" fontWeight="600">LAYER 2</text>
      <text x="16" y="280" fontSize="10" fill="var(--text-secondary)" opacity="0.5" fontWeight="600">LAYER 1</text>
      <text x="16" y="480" fontSize="10" fill="var(--text-secondary)" opacity="0.5" fontWeight="600">LAYER 0</text>

      {/* ─── DROP LINES: vertical dotted connectors from upper nodes to layer 0 ─── */}
      {dropLines.map(line => {
        const isVisited = visitedNodes.has(line.id)
        return (
          <line key={`drop-${line.id}`}
            x1={line.x} y1={line.fromY + 8} x2={line.x} y2={line.toY}
            stroke={isVisited ? 'var(--accent-yellow)' : '#252a40'}
            strokeWidth={isVisited ? 1.5 : 0.4}
            strokeDasharray="2 4"
            opacity={isVisited ? 0.7 : (isSearching ? 0.05 : 0.15)}
          />
        )
      })}

      {/* ─── UPPER LAYER EDGES (only between nodes in SAME visual band) ─── */}
      {upperEdges.map((edge, i) => {
        const source = nodePositions.get(edge.source)
        const target = nodePositions.get(edge.target)
        if (!source || !target) return null

        // Only draw if both nodes are in the SAME layer band visually
        const sourceNode = nodes.find(n => n.id === edge.source)
        const targetNode = nodes.find(n => n.id === edge.target)
        if (!sourceNode || !targetNode) return null
        if (sourceNode.level !== targetNode.level) return null  // skip cross-band edges

        const bothVisited = visitedNodes.has(edge.source) && visitedNodes.has(edge.target)

        return (
          <line key={`ue-${i}`}
            x1={source.x} y1={source.y} x2={target.x} y2={target.y}
            stroke={bothVisited ? '#ffdd44' : '#1e2845'}
            strokeWidth={bothVisited ? 2 : 0.4}
            opacity={bothVisited ? 0.85 : (isSearching ? 0.06 : 0.2)}
          />
        )
      })}

      {/* ─── LAYER 0 EDGES: only show visited during search ─── */}
      {isSearching && layer0Edges.map((edge, i) => {
        const source = nodePositions.get(edge.source)
        const target = nodePositions.get(edge.target)
        if (!source || !target) return null

        const bothVisited = visitedNodes.has(edge.source) && visitedNodes.has(edge.target)
        if (!bothVisited) return null

        return (
          <line key={`le-${i}`}
            x1={source.x} y1={source.y} x2={target.x} y2={target.y}
            stroke="#ffdd44" strokeWidth="1.5" opacity="0.6"
          />
        )
      })}

      {/* ─── NODES ─── */}
      {nodes.map(node => {
        const pos = nodePositions.get(node.id)
        if (!pos) return null

        const isResult = resultNodes.has(node.id)
        const isVisited = visitedNodes.has(node.id)

        // Size by layer
        let radius = node.level === 0 ? 3 : (node.level === 1 ? 5 : 7)
        let fill = node.level === 0 ? 'var(--node-default)' : (node.level === 1 ? '#6366f1' : 'var(--accent-purple)')
        let opacity = isSearching ? 0.12 : (node.level === 0 ? 0.45 : 0.7)

        if (isResult) {
          fill = 'var(--accent-green)'
          radius = 10
          opacity = 1
        } else if (isVisited) {
          fill = 'var(--accent-blue)'
          radius = Math.max(radius, 5)
          opacity = 0.9
        }

        return (
          <circle key={`n-${node.id}`}
            cx={pos.x} cy={pos.y} r={radius}
            fill={fill} opacity={opacity}
          >
            <title>{node.text || `Node ${node.id}`}</title>
          </circle>
        )
      })}

      {/* ─── RESULT LABELS ─── */}
      {Array.from(resultNodes).slice(0, 3).map((nodeId, i) => {
        const pos = nodePositions.get(nodeId)
        const node = nodes.find(n => n.id === nodeId)
        if (!pos || !node) return null

        return (
          <g key={`label-${nodeId}`}>
            <rect
              x={pos.x + 14} y={pos.y - 8 + i * 20}
              width={175} height={15}
              rx="3" fill="rgba(61, 214, 140, 0.12)"
              stroke="var(--accent-green)" strokeWidth="0.5"
            />
            <text
              x={pos.x + 18} y={pos.y + 3 + i * 20}
              fontSize="8.5" fontWeight="500"
              fill="var(--accent-green)"
            >
              {node.text.slice(0, 36)}...
            </text>
          </g>
        )
      })}
    </svg>
  )
}

export default GraphCanvas
