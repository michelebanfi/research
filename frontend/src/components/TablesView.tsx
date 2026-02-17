import { useState } from 'react'
import { Table, ChevronLeft, ChevronRight, Maximize2, Minimize2 } from 'lucide-react'

interface TableChunk {
  id: string
  content: string
  chunk_index: number
  metadata?: {
    page_number?: number
    headings?: string[]
  }
}

interface TablesViewProps {
  tables: TableChunk[]
}

export default function TablesView({ tables }: TablesViewProps) {
  const [selectedIndex, setSelectedIndex] = useState(0)
  const [isExpanded, setIsExpanded] = useState(false)
  
  if (tables.length === 0) {
    return (
      <div className="text-center py-12 text-muted">
        <Table size={48} className="mx-auto mb-4 opacity-50" />
        <p className="text-sm">No tables detected in this document</p>
      </div>
    )
  }
  
  const currentTable = tables[selectedIndex]
  
  return (
    <div className="h-full flex flex-col">
      {/* Header with navigation */}
      <div className="flex items-center justify-between px-4 py-3 bg-slate-50 dark:bg-slate-900/50 border-b border-border">
        <div className="flex items-center gap-3">
          <Table size={18} className="text-emerald-500" />
          <span className="font-medium">
            Table {selectedIndex + 1} of {tables.length}
          </span>
        </div>
        
        <div className="flex items-center gap-2">
          <button
            onClick={() => setSelectedIndex(Math.max(0, selectedIndex - 1))}
            disabled={selectedIndex === 0}
            className="p-1 rounded hover:bg-slate-200 dark:hover:bg-slate-800 disabled:opacity-30 transition-colors"
          >
            <ChevronLeft size={18} />
          </button>
          <button
            onClick={() => setSelectedIndex(Math.min(tables.length - 1, selectedIndex + 1))}
            disabled={selectedIndex === tables.length - 1}
            className="p-1 rounded hover:bg-slate-200 dark:hover:bg-slate-800 disabled:opacity-30 transition-colors"
          >
            <ChevronRight size={18} />
          </button>
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-1 rounded hover:bg-slate-200 dark:hover:bg-slate-800 transition-colors ml-2"
            title={isExpanded ? "Collapse" : "Expand"}
          >
            {isExpanded ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
          </button>
        </div>
      </div>
      
      {/* Table metadata */}
      <div className="px-4 py-2 border-b border-border bg-slate-50/50 dark:bg-slate-900/30 text-sm">
        <div className="flex items-center gap-4 text-xs text-muted">
          {currentTable.metadata?.page_number && (
            <span>Page {currentTable.metadata.page_number}</span>
          )}
          <span>Chunk #{currentTable.chunk_index}</span>
          <span className="text-slate-300">|</span>
          <span className="truncate max-w-md">
            {currentTable.metadata?.headings?.join(' > ') || 'No section context'}
          </span>
        </div>
      </div>
      
      {/* Table content */}
      <div className="flex-1 overflow-auto p-4">
        <div className={`
          bg-white dark:bg-slate-900 rounded-lg border border-border overflow-hidden
          ${isExpanded ? 'max-w-none' : 'max-w-3xl mx-auto'}
        `}>
          <div className="overflow-x-auto">
            <RenderMarkdownTable content={currentTable.content} />
          </div>
        </div>
      </div>
      
      {/* Table thumbnails grid */}
      {tables.length > 1 && (
        <div className="border-t border-border bg-slate-50 dark:bg-slate-900/50 p-3">
          <div className="text-xs text-muted mb-2">
            All tables ({tables.length})
          </div>
          <div className="flex gap-2 overflow-x-auto pb-1">
            {tables.map((table, idx) => (
              <button
                key={table.id}
                onClick={() => setSelectedIndex(idx)}
                className={`
                  flex-shrink-0 p-2 rounded border text-left transition-all
                  ${selectedIndex === idx 
                    ? 'border-emerald-500 bg-emerald-50 dark:bg-emerald-900/20' 
                    : 'border-border hover:border-slate-300 dark:hover:border-slate-600'}
                `}
                style={{ width: '120px' }}
              >
                <div className="flex items-center gap-1 text-xs text-muted mb-1">
                  <Table size={10} />
                  <span>#{table.chunk_index}</span>
                </div>
                <div className="text-xs truncate opacity-70">
                  {table.content.slice(0, 40).replace(/\|/g, ' ')}...
                </div>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function RenderMarkdownTable({ content }: { content: string }) {
  // Simple markdown table parser
  const lines = content.split('\n').filter(line => line.trim())
  
  if (lines.length < 2) {
    return <pre className="p-4 text-sm whitespace-pre-wrap">{content}</pre>
  }
  
  // Find table rows (lines with |)
  const tableLines = lines.filter(line => line.includes('|'))
  
  if (tableLines.length < 2) {
    return <pre className="p-4 text-sm whitespace-pre-wrap">{content}</pre>
  }
  
  // Parse header and rows
  const headerLine = tableLines[0]
  // Skip separator line (tableLines[1])
  const dataLines = tableLines.slice(2)
  
  const headers = headerLine.split('|').map(h => h.trim()).filter(h => h)
  
  return (
    <table className="w-full text-sm">
      <thead className="bg-slate-100 dark:bg-slate-800">
        <tr>
          {headers.map((header, idx) => (
            <th 
              key={idx} 
              className="px-4 py-2 text-left font-medium text-slate-700 dark:text-slate-300 border-b border-border"
            >
              {header}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {dataLines.map((line, rowIdx) => {
          const cells = line.split('|').map(c => c.trim()).filter(c => c)
          return (
            <tr 
              key={rowIdx}
              className="border-b border-border last:border-b-0 hover:bg-slate-50 dark:hover:bg-slate-800/50"
            >
              {cells.map((cell, cellIdx) => (
                <td key={cellIdx} className="px-4 py-2">
                  {cell}
                </td>
              ))}
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}
