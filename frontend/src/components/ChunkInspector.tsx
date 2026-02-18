import { useState } from 'react'
import { 
  X, 
  Copy, 
  Check,
  FileText,
  Table,
  BookOpen,
  Hash,
  Layers
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
    section_header?: string
  }
}

interface ChunkInspectorProps {
  chunk: Chunk | null
  onClose: () => void
  allChunks?: Chunk[]
}

export default function ChunkInspector({ chunk, onClose, allChunks = [] }: ChunkInspectorProps) {
  const [copied, setCopied] = useState(false)
  
  if (!chunk) return null
  
  const handleCopy = () => {
    navigator.clipboard.writeText(chunk.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }
  
  // Find parent chunk info
  const parentChunk = chunk.parent_chunk_id 
    ? allChunks.find(c => c.id === chunk.parent_chunk_id)
    : null
  
  return (
    <div className="w-80 border-l border-border bg-surface flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-border">
        <div className="flex items-center gap-2">
          {(chunk.is_table || chunk.metadata?.is_table) ? (
            <Table size={18} className="text-emerald-500" />
          ) : chunk.is_reference ? (
            <BookOpen size={18} className="text-amber-500" />
          ) : (
            <FileText size={18} className="text-blue-500" />
          )}
          <h3 className="font-semibold">Chunk Details</h3>
        </div>
        <button
          onClick={onClose}
          className="p-1 hover:bg-slate-200 dark:hover:bg-muted/20 rounded transition-colors"
        >
          <X size={18} />
        </button>
      </div>
      
      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Metadata */}
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-muted">ID:</span>
            <span className="font-mono text-xs">{chunk.id.slice(0, 16)}...</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted">Index:</span>
            <span className="flex items-center gap-1">
              <Hash size={12} />
              {chunk.chunk_index}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted">Level:</span>
            <span>{chunk.chunk_level}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted">Type:</span>
            <span>
              {(chunk.is_table || chunk.metadata?.is_table) ? 'Table' : chunk.is_reference ? 'Reference' : 'Text'}
            </span>
          </div>
          {chunk.metadata?.page_number && (
            <div className="flex justify-between">
              <span className="text-muted">Page:</span>
              <span>{chunk.metadata.page_number}</span>
            </div>
          )}
          {parentChunk && (
            <div className="flex justify-between">
              <span className="text-muted">Parent:</span>
              <span className="text-xs truncate max-w-32" title={parentChunk.id}>
                #{parentChunk.chunk_index}
              </span>
            </div>
          )}
        </div>
        
        {/* Section hierarchy */}
        {chunk.metadata?.headings && chunk.metadata.headings.length > 0 && (
          <div className="pt-2 border-t border-border">
            <p className="text-xs text-muted mb-2">Section Path:</p>
            <div className="space-y-1">
              {chunk.metadata.headings.map((heading, idx) => (
                <div 
                  key={idx}
                  className="flex items-center gap-2 text-sm"
                  style={{ paddingLeft: `${idx * 12}px` }}
                >
                  <Layers size={12} className="text-slate-400" />
                  <span className="truncate">{heading}</span>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {/* Content */}
        <div className="pt-2 border-t border-border">
          <div className="flex items-center justify-between mb-2">
            <p className="text-xs text-muted">Content:</p>
            <button
              onClick={handleCopy}
              className="flex items-center gap-1 text-xs text-secondary hover:text-secondary-hover transition-colors"
            >
              {copied ? (
                <>
                  <Check size={12} />
                  Copied!
                </>
              ) : (
                <>
                  <Copy size={12} />
                  Copy
                </>
              )}
            </button>
          </div>
          <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-3 text-sm whitespace-pre-wrap max-h-96 overflow-y-auto">
            {chunk.content}
          </div>
        </div>
        
        {/* Stats */}
        <div className="pt-2 border-t border-border">
          <p className="text-xs text-muted mb-2">Statistics:</p>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div className="bg-slate-50 dark:bg-slate-900 rounded p-2">
              <span className="text-muted text-xs">Characters:</span>
              <p className="font-medium">{chunk.content.length.toLocaleString()}</p>
            </div>
            <div className="bg-slate-50 dark:bg-slate-900 rounded p-2">
              <span className="text-muted text-xs">Words:</span>
              <p className="font-medium">{chunk.content.split(/\s+/).length.toLocaleString()}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
