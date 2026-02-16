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
      
      // Transform to ReactFlow format
      const flowNodes: Node[] = data.nodes.map((n: any, idx: number) => ({
        id: n.id,
        data: { label: n.label, type: n.type },
        position: { 
          x: 100 + (idx % 10) * 150, 
          y: 100 + Math.floor(idx / 10) * 100 
        },
        style: {
          background: n.color,
          color: 'white',
          border: 'none',
          padding: '10px',
          borderRadius: '8px',
          fontSize: '12px',
          fontWeight: 'bold',
        },
      }))
      
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
