import { useRef, useEffect, useMemo } from 'react'

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
  activeLayer,
  searchPath,
}: Props) {
  const svgRef = useRef<SVGSVGElement>(null)

  // Compute node positions mapped to SVG coordinates
  const nodePositions = useMemo(() => {
    const positions = new Map<number, { x: number; y: number; layer: number }>()
    if (!nodes.length) return positions

    const padding = 60
    const width = 900
    const height = 600

    // Separate layers vertically
    const layerHeight = (height - padding * 2) / (maxLevel + 1)

    nodes.forEach(node => {
      // x from PCA position [0,1] → pixel
      const x = padding + node.position[0] * (width - padding * 2)
      // y from layer (higher layers at top)
      const baseY = padding + (maxLevel - node.level) * layerHeight
      // Add some jitter based on PCA y-position within the layer band
      const jitter = node.position[1] * layerHeight * 0.7
      const y = baseY + jitter

      positions.set(node.id, { x, y, layer: node.level })
    })

    return positions
  }, [nodes, maxLevel])

  // Get node color based on animation state
  const getNodeColor = (nodeId: number) => {
    if (resultNodes.has(nodeId)) return 'var(--accent-green)'
    if (expandedNodes.has(nodeId)) return 'var(--accent-blue)'
    if (visitedNodes.has(nodeId)) return 'var(--accent-yellow)'
    return '#4a4f6b'
  }

  const getNodeRadius = (nodeId: number) => {
    if (resultNodes.has(nodeId)) return 7
    if (expandedNodes.has(nodeId)) return 5
    if (visitedNodes.has(nodeId)) return 4.5
    return 3
  }

  // Filter edges to only show layer 0 by default, or active layer during animation
  const visibleEdges = useMemo(() => {
    return edges.filter(e => {
      if (activeLayer !== null) return e.layer <= activeLayer
      return e.layer === 0
    })
  }, [edges, activeLayer])

  return (
    <svg ref={svgRef} viewBox="0 0 900 600" preserveAspectRatio="xMidYMid meet">
      {/* Background gradient */}
      <defs>
        <radialGradient id="bg-gradient">
          <stop offset="0%" stopColor="#1a1d2e" />
          <stop offset="100%" stopColor="#0f1117" />
        </radialGradient>
      </defs>
      <rect width="900" height="600" fill="url(#bg-gradient)" />

      {/* Layer separator lines */}
      {Array.from({ length: maxLevel + 1 }, (_, i) => {
        const y = 60 + (maxLevel - i) / (maxLevel + 1) * 480
        return (
          <line
            key={`layer-line-${i}`}
            x1="40"
            y1={y}
            x2="860"
            y2={y}
            stroke="var(--border)"
            strokeWidth="0.5"
            strokeDasharray="4 4"
            opacity="0.3"
          />
        )
      })}

      {/* Edges */}
      {visibleEdges.map((edge, i) => {
        const source = nodePositions.get(edge.source)
        const target = nodePositions.get(edge.target)
        if (!source || !target) return null

        const isHighlighted =
          visitedNodes.has(edge.source) && visitedNodes.has(edge.target)

        return (
          <line
            key={`edge-${i}`}
            x1={source.x}
            y1={source.y}
            x2={target.x}
            y2={target.y}
            stroke={
              isHighlighted
                ? 'var(--accent-yellow)'
                : edge.layer > 0
                ? 'var(--accent-purple)'
                : 'var(--border)'
            }
            strokeWidth={isHighlighted ? 1.5 : edge.layer > 0 ? 0.8 : 0.4}
            opacity={isHighlighted ? 0.8 : edge.layer > 0 ? 0.4 : 0.2}
          />
        )
      })}

      {/* Search path animation */}
      {searchPath.length >= 2 && (() => {
        const pathPoints: string[] = []
        for (let i = 0; i < searchPath.length; i += 2) {
          const from = nodePositions.get(searchPath[i])
          const to = nodePositions.get(searchPath[i + 1])
          if (from && to) {
            pathPoints.push(`M${from.x},${from.y} L${to.x},${to.y}`)
          }
        }
        return (
          <path
            d={pathPoints.join(' ')}
            className="search-path"
          />
        )
      })()}

      {/* Nodes */}
      {nodes.map(node => {
        const pos = nodePositions.get(node.id)
        if (!pos) return null

        return (
          <circle
            key={`node-${node.id}`}
            cx={pos.x}
            cy={pos.y}
            r={getNodeRadius(node.id)}
            fill={getNodeColor(node.id)}
            opacity={
              activeLayer !== null && node.level < activeLayer ? 0.2 : 0.9
            }
          >
            <title>{node.text || `Node ${node.id}`}</title>
          </circle>
        )
      })}

      {/* Result labels */}
      {Array.from(resultNodes).map(nodeId => {
        const pos = nodePositions.get(nodeId)
        const node = nodes.find(n => n.id === nodeId)
        if (!pos || !node) return null
        return (
          <text
            key={`label-${nodeId}`}
            x={pos.x + 10}
            y={pos.y + 4}
            fontSize="9"
            fill="var(--accent-green)"
            opacity="0.8"
          >
            {node.text.slice(0, 40)}...
          </text>
        )
      })}
    </svg>
  )
}

export default GraphCanvas
