import { useState, useEffect, useCallback } from 'react'
import { useAppStore } from '../stores/appStore'
import { api } from '../services/api'
import ReactFlow, {
  Node,
  Edge,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
} from 'reactflow'
import 'reactflow/dist/style.css'
import { Loader2, RefreshCw, Share2, X } from 'lucide-react'

// Force-directed layout algorithm for organic graph visualization
interface LayoutNode {
  id: string
  x: number
  y: number
  vx: number
  vy: number
}

function calculateForceDirectedLayout(
  nodes: Array<{ id: string }>,
  edges: Array<{ source: string; target: string }>,
  width: number = 1200,
  height: number = 800
): Map<string, { x: number; y: number }> {
  const nodeMap = new Map<string, LayoutNode>()
  const positions = new Map<string, { x: number; y: number }>()
  
  // Initialize nodes with random positions near center
  // Use larger initial radius for better distribution
  nodes.forEach((node, i) => {
    const angle = (i / nodes.length) * 2 * Math.PI
    const radius = Math.min(width, height) * 0.35  // Increased from 0.2
    nodeMap.set(node.id, {
      id: node.id,
      x: width / 2 + Math.cos(angle) * radius,
      y: height / 2 + Math.sin(angle) * radius,
      vx: 0,
      vy: 0
    })
  })
  
  // Force-directed simulation parameters
  const REPULSION_FORCE = 20000  // Increased from 5000 for more spacing
  const ATTRACTION_FORCE = 0.02  // Decreased from 0.05 for looser clusters
  const DAMPING = 0.85
  const MAX_DISPLACEMENT = 80  // Increased from 50
  const ITERATIONS = 150  // More iterations for better convergence
  const IDEAL_EDGE_LENGTH = 200  // Target distance between connected nodes
  
  // Run simulation
  for (let iter = 0; iter < ITERATIONS; iter++) {
    // Calculate repulsive forces (nodes push away from each other)
    nodeMap.forEach((node1) => {
      nodeMap.forEach((node2) => {
        if (node1.id === node2.id) return
        
        const dx = node1.x - node2.x
        const dy = node1.y - node2.y
        const distance = Math.sqrt(dx * dx + dy * dy) || 1
        
        const force = REPULSION_FORCE / (distance * distance)
        const fx = (dx / distance) * force
        const fy = (dy / distance) * force
        
        node1.vx += fx
        node1.vy += fy
      })
    })
    
    // Calculate attractive forces (edges pull nodes together)
    edges.forEach((edge) => {
      const source = nodeMap.get(edge.source)
      const target = nodeMap.get(edge.target)
      if (!source || !target) return
      
      const dx = target.x - source.x
      const dy = target.y - source.y
      const distance = Math.sqrt(dx * dx + dy * dy) || 1
      
      // Spring force: pulls toward ideal length
      const displacement = distance - IDEAL_EDGE_LENGTH
      const force = displacement * ATTRACTION_FORCE
      const fx = (dx / distance) * force
      const fy = (dy / distance) * force
      
      source.vx += fx
      source.vy += fy
      target.vx -= fx
      target.vy -= fy
    })
    
    // Apply forces with damping and center gravity
    const CENTER_GRAVITY = 0.01
    nodeMap.forEach((node) => {
      // Center gravity (pull toward center)
      node.vx += (width / 2 - node.x) * CENTER_GRAVITY
      node.vy += (height / 2 - node.y) * CENTER_GRAVITY
      
      // Apply damping
      node.vx *= DAMPING
      node.vy *= DAMPING
      
      // Limit maximum displacement
      const velocity = Math.sqrt(node.vx * node.vx + node.vy * node.vy)
      if (velocity > MAX_DISPLACEMENT) {
        node.vx = (node.vx / velocity) * MAX_DISPLACEMENT
        node.vy = (node.vy / velocity) * MAX_DISPLACEMENT
      }
      
      // Update position
      node.x += node.vx
      node.y += node.vy
      
      // Keep within bounds with padding
      const padding = 150  // Increased from 100 for better margins
      node.x = Math.max(padding, Math.min(width - padding, node.x))
      node.y = Math.max(padding, Math.min(height - padding, node.y))
    })
  }
  
  // Extract final positions
  nodeMap.forEach((node) => {
    positions.set(node.id, { x: node.x, y: node.y })
  })
  
  return positions
}

export default function GraphTab() {
  const [isLoading, setIsLoading] = useState(false)
  const [nodes, setNodes, onNodesChange] = useNodesState([])
  const [edges, setEdges, onEdgesChange] = useEdgesState([])
  const [selectedNode, setSelectedNode] = useState<Node | null>(null)
  
  const { selectedProject } = useAppStore()

  const loadGraph = useCallback(async () => {
    if (!selectedProject) return
    
    setIsLoading(true)
    try {
      const data = await api.getProjectGraph(selectedProject.id)
      
      // Calculate force-directed layout for organic graph visualization
      // Using larger canvas for better spacing
      const layoutPositions = calculateForceDirectedLayout(
        data.nodes,
        data.edges,
        1600,
        1200
      )
      
      // Transform to ReactFlow format with calculated positions
      const flowNodes: Node[] = data.nodes.map((n: any) => {
        const position = layoutPositions.get(n.id) || { x: 600, y: 400 }
        return {
          id: n.id,
          data: { label: n.label, type: n.type },
          position,
          style: {
            background: n.color,
            color: 'white',
            border: 'none',
            padding: '10px',
            borderRadius: '8px',
            fontSize: '12px',
            fontWeight: 'bold',
          },
        }
      })
      
      const flowEdges: Edge[] = data.edges.map((e: any, idx: number) => ({
        id: `e${idx}`,
        source: e.source,
        target: e.target,
        label: e.label,
        animated: true,
        style: { stroke: '#6366f1', strokeWidth: 2 },
        labelStyle: { fill: '#818cf8', fontSize: 10 },
      }))
      
      setNodes(flowNodes)
      setEdges(flowEdges)
    } catch (error) {
      console.error('Failed to load graph:', error)
    } finally {
      setIsLoading(false)
    }
  }, [selectedProject, setNodes, setEdges])

  useEffect(() => {
    loadGraph()
  }, [loadGraph])

  const onNodeClick = (_: React.MouseEvent, node: Node) => {
    setSelectedNode(node)
  }

  return (
    <div className="flex h-full">
      {/* Graph Area */}
      <div className="flex-1 flex flex-col">
        <div className="flex items-center justify-between px-4 py-2 border-b border-border">
          <div className="flex items-center gap-2">
            <Share2 size={20} />
            <div>
              <h2 className="text-lg font-semibold">Knowledge Graph</h2>
              <p className="text-xs text-muted">
                {nodes.length} nodes, {edges.length} edges
              </p>
            </div>
          </div>
          <button
            onClick={loadGraph}
            disabled={isLoading}
            className="flex items-center gap-2 px-4 py-2 bg-secondary text-white rounded-lg hover:bg-secondary-hover disabled:opacity-50 transition-colors"
          >
            {isLoading ? (
              <Loader2 size={18} className="animate-spin" />
            ) : (
              <RefreshCw size={18} />
            )}
            Refresh
          </button>
        </div>

        <div className="flex-1 relative">
          {nodes.length === 0 && !isLoading ? (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center">
                <p className="text-muted mb-2">No graph data yet</p>
                <p className="text-sm text-muted/60">
                  Ingest some files to build the knowledge graph
                </p>
              </div>
            </div>
          ) : (
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onNodeClick={onNodeClick}
              fitView
              attributionPosition="bottom-left"
            >
              <Background color="#94a3b8" gap={16} />
              <Controls />
              <MiniMap 
                nodeStrokeWidth={3}
                zoomable
                pannable
              />
            </ReactFlow>
          )}
        </div>

        {/* Legend */}
        <div className="px-4 py-2 bg-surface border-t border-border">
          <div className="flex items-center gap-4 text-xs">
            <span className="font-medium">Legend:</span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 rounded-full bg-[#FF6B6B]"></span> Concept
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 rounded-full bg-[#4ECDC4]"></span> Tool
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 rounded-full bg-[#45B7D1]"></span> System
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 rounded-full bg-[#96CEB4]"></span> Metric
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 rounded-full bg-[#DDA0DD]"></span> Person
            </span>
          </div>
        </div>
      </div>

      {/* Selected Node Info */}
      {selectedNode && (
        <div className="w-72 border-l border-border bg-surface/50 p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold">Node Details</h3>
            <button
              onClick={() => setSelectedNode(null)}
              className="p-1 hover:bg-slate-200 dark:hover:bg-muted/20 rounded transition-colors"
            >
              <X size={16} />
            </button>
          </div>
          <div className="space-y-3">
            <div>
              <p className="text-xs text-muted">Name</p>
              <p className="text-sm font-medium">{selectedNode.data.label}</p>
            </div>
            <div>
              <p className="text-xs text-muted">Type</p>
              <p className="text-sm">{selectedNode.data.type}</p>
            </div>
            <div>
              <p className="text-xs text-muted">Connections</p>
              <p className="text-sm">
                {edges.filter(e => e.source === selectedNode.id || e.target === selectedNode.id).length} edges
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
