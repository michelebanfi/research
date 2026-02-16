import { useState, useRef } from 'react'
import { useAppStore } from '../stores/appStore'
import { api } from '../services/api'
import { Upload, File, Trash2, Loader2, FolderOpen } from 'lucide-react'

export default function IngestTab() {
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)
  
  const { 
    selectedProject, 
    files, 
    setFiles 
  } = useAppStore()

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file || !selectedProject) return

    setIsUploading(true)
    setUploadProgress('Uploading file...')

    try {
      const result = await api.uploadFile(file, selectedProject.id)
      
      if (result.success) {
        setUploadProgress(`Success! Processed ${result.chunks_count} chunks, ${result.nodes_count} nodes, ${result.edges_count} edges`)
        // Refresh file list
        const updatedFiles = await api.getProjectFiles(selectedProject.id)
        setFiles(updatedFiles)
      } else {
        setUploadProgress(`Error: ${result.message}`)
      }
    } catch (error) {
      console.error('Upload error:', error)
      setUploadProgress('Error uploading file')
    } finally {
      setIsUploading(false)
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    }
  }

  const handleDelete = async (fileId: string) => {
    if (!confirm('Are you sure you want to delete this file?')) return
    
    try {
      await api.deleteFile(fileId)
      // Refresh file list
      if (selectedProject) {
        const updatedFiles = await api.getProjectFiles(selectedProject.id)
        setFiles(updatedFiles)
      }
    } catch (error) {
      console.error('Delete error:', error)
      alert('Failed to delete file')
    }
  }

  return (
    <div className="flex h-full gap-4 p-4">
      {/* Left - File List */}
      <div className="flex-1 flex flex-col min-w-0">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <FolderOpen size={20} />
            <h2 className="text-lg font-semibold">Project Files</h2>
          </div>
          <span className="text-sm text-muted">
            {files.length} file{files.length !== 1 ? 's' : ''}
          </span>
        </div>

        <div className="flex-1 overflow-y-auto space-y-2">
          {files.length === 0 ? (
            <div className="text-center py-12">
              <File size={48} className="mx-auto text-muted mb-4" />
              <p className="text-muted">No files ingested yet</p>
              <p className="text-sm text-muted/60 mt-1">
                Upload documents or code files to get started
              </p>
            </div>
          ) : (
            files.map((file) => (
              <FileCard 
                key={file.id} 
                file={file} 
                onDelete={() => handleDelete(file.id)}
              />
            ))
          )}
        </div>
      </div>

      {/* Right - Upload Area */}
      <div className="w-96 flex flex-col">
        <div className="bg-surface rounded-lg border border-border p-6 shadow-sm">
          <div className="flex items-center gap-2 mb-4">
            <Upload size={20} />
            <h3 className="text-lg font-semibold">Upload New File</h3>
          </div>
          
          <div className="border-2 border-dashed border-border rounded-lg p-8 text-center hover:border-secondary/50 transition-colors bg-slate-50/50 dark:bg-transparent">
            <input
              ref={fileInputRef}
              type="file"
              onChange={handleFileSelect}
              accept=".pdf,.md,.txt,.py,.docx"
              disabled={isUploading}
              className="hidden"
              id="file-upload"
            />
            <label
              htmlFor="file-upload"
              className="cursor-pointer flex flex-col items-center"
            >
              {isUploading ? (
                <>
                  <Loader2 size={48} className="text-secondary animate-spin mb-4" />
                  <p className="text-secondary font-medium">Processing...</p>
                  <p className="text-sm text-muted mt-2">{uploadProgress}</p>
                </>
              ) : (
                <>
                  <Upload size={48} className="text-muted mb-4" />
                  <p className="text-muted font-medium">
                    Click to upload or drag and drop
                  </p>
                  <p className="text-sm text-muted/60 mt-2">
                    PDF, MD, TXT, PY, DOCX
                  </p>
                </>
              )}
            </label>
          </div>

          {!isUploading && uploadProgress && (
            <div className={`mt-4 p-3 rounded-lg text-sm ${
              uploadProgress.includes('Error') 
                ? 'bg-red-100 text-red-700 dark:bg-red-500/10 dark:text-red-400' 
                : 'bg-emerald-100 text-emerald-700 dark:bg-green-500/10 dark:text-green-400'
            }`}>
              {uploadProgress}
            </div>
          )}

          <div className="mt-6 space-y-2 text-sm text-muted">
            <p className="font-medium text-text">Supported formats:</p>
            <ul className="space-y-1 ml-4">
              <li>• PDF documents</li>
              <li>• Markdown files</li>
              <li>• Python code</li>
              <li>• Text files</li>
              <li>• Word documents</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

function FileCard({ file, onDelete }: { file: any; onDelete: () => void }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="bg-surface rounded-lg border border-border overflow-hidden shadow-sm">
      <div className="p-4 flex items-start justify-between">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <File size={18} className="text-secondary" />
            <h4 className="font-medium truncate">{file.name}</h4>
          </div>
          <p className="text-xs text-muted mt-1">
            Processed: {new Date(file.processed_at).toLocaleString()}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setExpanded(!expanded)}
            className="px-3 py-1 text-sm bg-slate-100 dark:bg-muted/20 rounded hover:bg-slate-200 dark:hover:bg-muted/30 transition-colors border border-border"
          >
            {expanded ? 'Hide' : 'Details'}
          </button>
          <button
            onClick={onDelete}
            className="p-2 text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-500/10 rounded transition-colors"
            title="Delete file"
          >
            <Trash2 size={18} />
          </button>
        </div>
      </div>

      {expanded && (
        <div className="px-4 pb-4 border-t border-border pt-3 bg-slate-50/50 dark:bg-transparent">
          <div className="space-y-3">
            <div>
              <p className="text-xs text-muted mb-1">Summary:</p>
              <p className="text-sm">{file.summary || 'No summary available'}</p>
            </div>
            
            {file.metadata?.keywords && file.metadata.keywords.length > 0 && (
              <div>
                <p className="text-xs text-muted mb-1">Keywords:</p>
                <div className="flex flex-wrap gap-1">
                  {file.metadata.keywords.slice(0, 10).map((keyword: string, idx: number) => (
                    <span
                      key={idx}
                      className="px-2 py-0.5 bg-slate-200 text-slate-700 dark:bg-secondary/20 dark:text-secondary text-xs rounded font-medium"
                    >
                      {keyword}
                    </span>
                  ))}
                </div>
              </div>
            )}
            
            <div className="text-xs text-muted">
              <p>ID: {file.id}</p>
              <p className="truncate">Path: {file.path}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
