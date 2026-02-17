import { useState, useRef, useEffect, useCallback } from 'react'
import { useAppStore } from '../stores/appStore'
import { api } from '../services/api'
import { Upload, File, Loader2, FolderOpen, CheckCircle2 } from 'lucide-react'
import EnhancedFileCard from './EnhancedFileCard'

interface IngestionProgress {
  file_id: string
  stage: string
  progress: number
  message: string
  chunks_count: number
  nodes_count: number
  edges_count: number
}

const STAGE_LABELS: Record<string, string> = {
  'saving': 'Saving file...',
  'parsing': 'Parsing document...',
  'summarizing': 'Generating summary...',
  'extracting_graph': 'Extracting entities...',
  'embedding': 'Creating embeddings...',
  'storing': 'Storing data...',
  'complete': 'Complete!'
}

export default function IngestTab() {
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState<IngestionProgress | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  
  const { 
    selectedProject, 
    files, 
    setFiles 
  } = useAppStore()

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current)
      }
    }
  }, [])

  const pollProgress = useCallback(async (tempId: string) => {
    try {
      const progress = await api.getUploadProgress(tempId)
      setUploadProgress(progress)
      
      // Stop polling if complete or error
      if (progress.stage === 'complete' || progress.progress >= 100) {
        if (pollingRef.current) {
          clearInterval(pollingRef.current)
          pollingRef.current = null
        }
      }
    } catch (error) {
      console.error('Error polling progress:', error)
    }
  }, [])

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file || !selectedProject) return

    setIsUploading(true)
    setUploadError(null)
    setUploadProgress(null)
    
    try {
      const result = await api.uploadFile(file, selectedProject.id)
      
      if (result.success && result.temp_file_id) {
        // Start polling for progress
        pollingRef.current = setInterval(() => {
          pollProgress(result.temp_file_id)
        }, 500) // Poll every 500ms
        
        // Initial poll
        pollProgress(result.temp_file_id)
        
        // Stop polling after 60 seconds (timeout) and refresh files
        setTimeout(() => {
          if (pollingRef.current) {
            clearInterval(pollingRef.current)
            pollingRef.current = null
          }
          // Refresh file list
          api.getProjectFiles(selectedProject.id).then(setFiles)
          setIsUploading(false)
          setUploadProgress(null)
        }, 60000)
        
      } else if (result.success) {
        // No progress tracking available, just refresh
        setUploadProgress({
          file_id: '',
          stage: 'complete',
          progress: 100,
          message: result.message,
          chunks_count: result.chunks_count,
          nodes_count: result.nodes_count,
          edges_count: result.edges_count
        })
        
        const updatedFiles = await api.getProjectFiles(selectedProject.id)
        setFiles(updatedFiles)
        setIsUploading(false)
        
        setTimeout(() => {
          setUploadProgress(null)
        }, 3000)
      } else {
        setUploadError(result.message)
        setIsUploading(false)
      }
    } catch (error) {
      console.error('Upload error:', error)
      setUploadError('Error uploading file')
      setIsUploading(false)
      if (pollingRef.current) {
        clearInterval(pollingRef.current)
        pollingRef.current = null
      }
    } finally {
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

        <div className="flex-1 overflow-y-auto space-y-3">
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
              <EnhancedFileCard 
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

          {/* Progress Bar */}
          {isUploading && uploadProgress && (
            <div className="mt-4 space-y-3">
              {/* Stage message */}
              <div className="flex items-center justify-between text-sm">
                <span className="text-secondary font-medium">
                  {STAGE_LABELS[uploadProgress.stage] || uploadProgress.message}
                </span>
                <span className="text-muted">
                  {Math.round(uploadProgress.progress)}%
                </span>
              </div>
              
              {/* Progress bar */}
              <div className="h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-secondary transition-all duration-300 ease-out"
                  style={{ width: `${uploadProgress.progress}%` }}
                />
              </div>
              
              {/* Stats */}
              {uploadProgress.chunks_count > 0 && (
                <div className="flex items-center gap-3 text-xs text-muted">
                  <span>{uploadProgress.chunks_count} chunks</span>
                  {uploadProgress.nodes_count > 0 && (
                    <>
                      <span>•</span>
                      <span>{uploadProgress.nodes_count} entities</span>
                    </>
                  )}
                  {uploadProgress.edges_count > 0 && (
                    <>
                      <span>•</span>
                      <span>{uploadProgress.edges_count} relations</span>
                    </>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Success message */}
          {!isUploading && uploadProgress?.stage === 'complete' && (
            <div className="mt-4 p-3 rounded-lg text-sm bg-emerald-100 text-emerald-700 dark:bg-green-500/10 dark:text-green-400 flex items-center gap-2">
              <CheckCircle2 size={18} />
              <span>
                Success! Processed {uploadProgress.chunks_count} chunks,{' '}
                {uploadProgress.nodes_count} nodes, {uploadProgress.edges_count} edges
              </span>
            </div>
          )}

          {/* Error message */}
          {!isUploading && uploadError && (
            <div className="mt-4 p-3 rounded-lg text-sm bg-red-100 text-red-700 dark:bg-red-500/10 dark:text-red-400">
              {uploadError}
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
