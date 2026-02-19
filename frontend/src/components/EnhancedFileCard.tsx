import { useState, useEffect } from 'react'
import {
  FileText,
  Trash2,
  ChevronDown,
  ChevronUp,
  FolderOpen,
  Table,
  Share2,
  BookOpen,
  Layers,
  Loader2,
  AlertCircle
} from 'lucide-react'
import { api } from '../services/api'
import FileStructureTree from './FileStructureTree'
import TablesView from './TablesView'
import EntitiesView from './EntitiesView'
import ChunkInspector from './ChunkInspector'

interface FileData {
  id: string
  name: string
  path: string
  summary?: string
  processed_at: string
  metadata?: {
    keywords?: string[]
  }
}

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
    is_table?: boolean
  }
}

interface Entity {
  name: string
  type: string
}

interface FileDetails {
  file: FileData
  chunks: Chunk[]
  sections: any[]
  keywords: string[]
  entities: Entity[]
  stats: {
    total_chunks: number
    table_chunks: number
    reference_chunks: number
    parent_chunks: number
    leaf_chunks: number
  }
}

interface EnhancedFileCardProps {
  file: FileData
  onDelete: () => void
}

type TabType = 'structure' | 'tables' | 'entities' | 'summary'

export default function EnhancedFileCard({ file, onDelete }: EnhancedFileCardProps) {
  const [isExpanded, setIsExpanded] = useState(false)
  const [activeTab, setActiveTab] = useState<TabType>('structure')
  const [details, setDetails] = useState<FileDetails | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedChunk, setSelectedChunk] = useState<Chunk | null>(null)
  const [highlightedEntity, setHighlightedEntity] = useState<string | null>(null)

  // Load file details when expanded
  useEffect(() => {
    if (isExpanded && !details && !isLoading) {
      loadDetails()
    }
  }, [isExpanded])

  const loadDetails = async () => {
    setIsLoading(true)
    setError(null)
    try {
      const data = await api.getFileDetails(file.id)
      setDetails(data)
    } catch (err) {
      setError('Failed to load file details')
      console.error('Error loading file details:', err)
    } finally {
      setIsLoading(false)
    }
  }

  const handleChunkSelect = (chunk: Chunk) => {
    setSelectedChunk(chunk)
  }

  const handleEntityClick = (entityName: string) => {
    setHighlightedEntity(entityName)
    setActiveTab('structure')
    // Clear highlight after 3 seconds
    setTimeout(() => setHighlightedEntity(null), 3000)
  }

  const tableChunks = details?.chunks.filter(c => c.is_table || c.metadata?.is_table) || []

  return (
    <div className="bg-surface rounded-lg border border-border overflow-hidden shadow-sm">
      {/* Header */}
      <div className="p-4 flex items-start justify-between">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <FileText size={18} className="text-secondary" />
            <h4 className="font-medium truncate">{file.name}</h4>
          </div>
          <p className="text-xs text-muted mt-1">
            Processed: {new Date(file.processed_at).toLocaleString()}
          </p>

          {/* Quick stats row */}
          {details && (
            <div className="flex items-center gap-3 mt-2 text-xs">
              <span className="flex items-center gap-1 text-slate-500">
                <Layers size={12} />
                {details.stats.total_chunks} chunks
              </span>
              {details.stats.table_chunks > 0 && (
                <span className="flex items-center gap-1 text-emerald-600">
                  <Table size={12} />
                  {details.stats.table_chunks} tables
                </span>
              )}
              {details.stats.reference_chunks > 0 && (
                <span className="flex items-center gap-1 text-amber-600">
                  <BookOpen size={12} />
                  {details.stats.reference_chunks} refs
                </span>
              )}
              {details.keywords.length > 0 && (
                <span className="flex items-center gap-1 text-purple-600">
                  <Share2 size={12} />
                  {details.keywords.length} entities
                </span>
              )}
            </div>
          )}
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="flex items-center gap-1 px-3 py-1.5 text-sm bg-slate-100 dark:bg-muted/20 rounded-lg hover:bg-slate-200 dark:hover:bg-muted/30 transition-colors border border-border"
          >
            {isExpanded ? (
              <>
                <ChevronUp size={16} />
                Collapse
              </>
            ) : (
              <>
                <ChevronDown size={16} />
                Details
              </>
            )}
          </button>
          <button
            onClick={onDelete}
            className="p-2 text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-500/10 rounded-lg transition-colors"
            title="Delete file"
          >
            <Trash2 size={18} />
          </button>
        </div>
      </div>

      {/* Expanded content */}
      {isExpanded && (
        <div className="border-t border-border">
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 size={24} className="text-secondary animate-spin" />
              <span className="ml-2 text-muted">Loading file details...</span>
            </div>
          ) : error ? (
            <div className="flex items-center justify-center py-12 text-red-500">
              <AlertCircle size={20} className="mr-2" />
              <span>{error}</span>
              <button
                onClick={loadDetails}
                className="ml-4 text-sm underline"
              >
                Retry
              </button>
            </div>
          ) : details ? (
            <div className="flex" style={{ height: '500px' }}>
              {/* Main content area */}
              <div className="flex-1 flex flex-col min-w-0">
                {/* Tabs */}
                <div className="flex border-b border-border bg-slate-50 dark:bg-slate-900/50">
                  <TabButton
                    active={activeTab === 'structure'}
                    onClick={() => setActiveTab('structure')}
                    icon={<FolderOpen size={16} />}
                    label="Structure"
                    count={details.stats.total_chunks}
                  />
                  <TabButton
                    active={activeTab === 'tables'}
                    onClick={() => setActiveTab('tables')}
                    icon={<Table size={16} />}
                    label="Tables"
                    count={details.stats.table_chunks}
                  />
                  <TabButton
                    active={activeTab === 'entities'}
                    onClick={() => setActiveTab('entities')}
                    icon={<Share2 size={16} />}
                    label="Entities"
                    count={details.keywords.length}
                  />
                  <TabButton
                    active={activeTab === 'summary'}
                    onClick={() => setActiveTab('summary')}
                    icon={<FileText size={16} />}
                    label="Summary"
                  />
                </div>

                {/* Tab content */}
                <div className="flex-1 overflow-hidden">
                  {activeTab === 'structure' && (
                    <FileStructureTree
                      chunks={details.chunks}
                      selectedChunkId={selectedChunk?.id || null}
                      onChunkSelect={handleChunkSelect}
                      highlightedEntity={highlightedEntity}
                    />
                  )}

                  {activeTab === 'tables' && (
                    <TablesView tables={tableChunks} />
                  )}

                  {activeTab === 'entities' && (
                    <EntitiesView
                      entities={details.entities}
                    />
                  )}

                  {activeTab === 'summary' && (
                    <div className="p-6 overflow-y-auto h-full">
                      <div className="max-w-3xl">
                        <h5 className="font-medium mb-3">Document Summary</h5>
                        <p className="text-sm text-slate-700 dark:text-slate-300 leading-relaxed">
                          {file.summary || 'No summary available'}
                        </p>

                        {file.metadata?.keywords && file.metadata.keywords.length > 0 && (
                          <div className="mt-6">
                            <h6 className="text-sm font-medium mb-2">Keywords</h6>
                            <div className="flex flex-wrap gap-2">
                              {file.metadata.keywords.slice(0, 20).map((keyword, idx) => (
                                <span
                                  key={idx}
                                  className="px-2 py-1 bg-slate-200 text-slate-700 dark:bg-secondary/20 dark:text-secondary text-xs rounded font-medium cursor-pointer hover:bg-slate-300 dark:hover:bg-secondary/30 transition-colors"
                                  onClick={() => handleEntityClick(keyword)}
                                >
                                  {keyword}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}

                        <div className="mt-6 pt-4 border-t border-border">
                          <h6 className="text-sm font-medium mb-2">File Information</h6>
                          <div className="space-y-1 text-xs text-muted">
                            <p><span className="font-medium">ID:</span> {file.id}</p>
                            <p><span className="font-medium">Path:</span> <span className="truncate">{file.path}</span></p>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Chunk Inspector Panel */}
              {selectedChunk && (
                <ChunkInspector
                  chunk={selectedChunk}
                  onClose={() => setSelectedChunk(null)}
                  allChunks={details.chunks}
                />
              )}
            </div>
          ) : null}
        </div>
      )}
    </div>
  )
}

interface TabButtonProps {
  active: boolean
  onClick: () => void
  icon: React.ReactNode
  label: string
  count?: number
}

function TabButton({ active, onClick, icon, label, count }: TabButtonProps) {
  return (
    <button
      onClick={onClick}
      className={`
        flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors
        ${active
          ? 'text-secondary border-b-2 border-secondary bg-white dark:bg-slate-800'
          : 'text-muted hover:text-text hover:bg-slate-100 dark:hover:bg-slate-800/50'
        }
      `}
    >
      {icon}
      <span>{label}</span>
      {count !== undefined && count > 0 && (
        <span className={`
          ml-1 px-2 py-0.5 text-xs rounded-full
          ${active
            ? 'bg-secondary/10 text-secondary'
            : 'bg-slate-200 text-slate-600 dark:bg-slate-700 dark:text-slate-400'
          }
        `}>
          {count}
        </span>
      )}
    </button>
  )
}
