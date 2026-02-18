import { useState, useMemo } from 'react'
import { 
  FolderOpen, 
  FileText, 
  Table, 
  BookOpen, 
  ChevronRight, 
  ChevronDown,
  Layers,
  Hash
} from 'lucide-react'

interface Chunk {
  id: string
  content: string
  chunk_index: number
  chunk_level: number
  parent_chunk_id: string | null
  is_table: boolean
  is_reference: boolean
  metadata?: {
    headings?: string[]
    doc_item_type?: string
    page_number?: number
  }
}

interface TreeNode extends Chunk {
  children: TreeNode[]
}

interface FileStructureTreeProps {
  chunks: Chunk[]
  selectedChunkId?: string | null
  onChunkSelect?: (chunk: Chunk) => void
  highlightedEntity?: string | null
}

function buildChunkTree(chunks: Chunk[]): TreeNode[] {
  const chunkMap = new Map<string, TreeNode>()
  
  // First pass: create nodes
  chunks.forEach(chunk => {
    chunkMap.set(chunk.id, {
      ...chunk,
      children: []
    })
  })
  
  // Second pass: build parent-child relationships
  const roots: TreeNode[] = []
  chunks.forEach(chunk => {
    const node = chunkMap.get(chunk.id)!
    if (chunk.parent_chunk_id && chunkMap.has(chunk.parent_chunk_id)) {
      const parent = chunkMap.get(chunk.parent_chunk_id)!
      parent.children.push(node)
    } else {
      roots.push(node)
    }
  })
  
  // Sort by chunk_index
  roots.sort((a, b) => (a.chunk_index || 0) - (b.chunk_index || 0))
  roots.forEach(root => {
    root.children.sort((a, b) => (a.chunk_index || 0) - (b.chunk_index || 0))
  })
  
  return roots
}

function getChunkIcon(chunk: Chunk, isExpanded: boolean, hasChildren?: boolean) {
  if (chunk.is_reference) {
    return <BookOpen size={14} className="text-amber-500" />
  }
  if (chunk.is_table || chunk.metadata?.is_table) {
    return <Table size={14} className="text-emerald-500" />
  }
  if (hasChildren) {
    return isExpanded 
      ? <FolderOpen size={14} className="text-blue-500" />
      : <FolderOpen size={14} className="text-blue-500" />
  }
  return <FileText size={14} className="text-slate-400" />
}

function TreeNodeComponent({ 
  node, 
  level, 
  selectedChunkId, 
  onChunkSelect,
  highlightedEntity,
  defaultExpanded = false
}: { 
  node: TreeNode
  level: number
  selectedChunkId?: string | null
  onChunkSelect?: (chunk: Chunk) => void
  highlightedEntity?: string | null
  defaultExpanded?: boolean
}) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded || level < 2)
  const hasChildren = node.children && node.children.length > 0
  const isSelected = selectedChunkId === node.id
  
  // Check if this chunk should be highlighted based on entity
  const shouldHighlight = highlightedEntity && 
    node.content.toLowerCase().includes(highlightedEntity.toLowerCase())
  
  const content = node.content || ''
  const preview = content.slice(0, 60).replace(/\n/g, ' ')
  const displayText = preview.length < content.length ? preview + '...' : preview
  
  return (
    <div className="select-none">
      <div 
        className={`
          flex items-center gap-1 py-1 px-2 rounded cursor-pointer
          transition-colors duration-150
          ${isSelected ? 'bg-blue-100 dark:bg-blue-900/30' : 'hover:bg-slate-100 dark:hover:bg-slate-800'}
          ${shouldHighlight ? 'ring-1 ring-amber-400 bg-amber-50 dark:bg-amber-900/20' : ''}
        `}
        style={{ paddingLeft: `${level * 16 + 8}px` }}
        onClick={() => {
          if (hasChildren) {
            setIsExpanded(!isExpanded)
          }
          onChunkSelect?.(node)
        }}
      >
        {hasChildren && (
          <span className="text-slate-400">
            {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          </span>
        )}
        {!hasChildren && <span className="w-[14px]" />}
        
        {getChunkIcon(node, isExpanded, hasChildren)}
        
        <span className="text-xs text-slate-500 ml-1">
          #{node.chunk_index}
        </span>
        
        <span className="text-sm truncate flex-1 ml-2" title={content}>
          {(node.is_table || node.metadata?.is_table) ? '[Table] ' : ''}
          {node.is_reference ? '[Ref] ' : ''}
          {displayText || 'Empty chunk'}
        </span>
        
        {node.metadata?.page_number && (
          <span className="text-xs text-slate-400 ml-2">
            p.{node.metadata.page_number}
          </span>
        )}
      </div>
      
      {isExpanded && hasChildren && (
        <div>
          {node.children.map(child => (
            <TreeNodeComponent
              key={child.id}
              node={child}
              level={level + 1}
              selectedChunkId={selectedChunkId}
              onChunkSelect={onChunkSelect}
              highlightedEntity={highlightedEntity}
            />
          ))}
        </div>
      )}
    </div>
  )
}

export default function FileStructureTree({ 
  chunks, 
  selectedChunkId,
  onChunkSelect,
  highlightedEntity
}: FileStructureTreeProps) {
  const tree = useMemo(() => buildChunkTree(chunks), [chunks])
  
  const stats = useMemo(() => ({
    total: chunks.length,
    tables: chunks.filter(c => c.is_table || c.metadata?.is_table).length,
    references: chunks.filter(c => c.is_reference).length,
    parents: chunks.filter(c => c.chunk_level === 0).length,
    leaves: chunks.filter(c => c.chunk_level > 0).length,
  }), [chunks])
  
  if (chunks.length === 0) {
    return (
      <div className="text-center py-8 text-muted">
        <Layers size={32} className="mx-auto mb-2 opacity-50" />
        <p className="text-sm">No chunks available</p>
      </div>
    )
  }
  
  return (
    <div className="h-full flex flex-col">
      {/* Stats Header */}
      <div className="flex items-center gap-3 px-3 py-2 bg-slate-50 dark:bg-slate-900/50 border-b border-border text-xs">
        <div className="flex items-center gap-1">
          <Hash size={12} />
          <span className="font-medium">{stats.total}</span>
          <span className="text-muted">total</span>
        </div>
        <div className="flex items-center gap-1">
          <FolderOpen size={12} className="text-blue-500" />
          <span className="font-medium">{stats.parents}</span>
          <span className="text-muted">sections</span>
        </div>
        <div className="flex items-center gap-1">
          <Table size={12} className="text-emerald-500" />
          <span className="font-medium">{stats.tables}</span>
          <span className="text-muted">tables</span>
        </div>
        <div className="flex items-center gap-1">
          <BookOpen size={12} className="text-amber-500" />
          <span className="font-medium">{stats.references}</span>
          <span className="text-muted">refs</span>
        </div>
      </div>
      
      {/* Tree Content */}
      <div className="flex-1 overflow-y-auto py-2">
        {tree.map(node => (
          <TreeNodeComponent
            key={node.id}
            node={node}
            level={0}
            selectedChunkId={selectedChunkId}
            onChunkSelect={onChunkSelect}
            highlightedEntity={highlightedEntity}
            defaultExpanded={true}
          />
        ))}
      </div>
    </div>
  )
}
