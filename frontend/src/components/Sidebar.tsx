import { useState } from 'react'
import { useAppStore } from '../stores/appStore'
import { api } from '../services/api'
import { Folder, Plus, Moon, Sun, Cpu } from 'lucide-react'

export default function Sidebar() {
  const [isCreating, setIsCreating] = useState(false)
  const [newProjectName, setNewProjectName] = useState('')
  const [isDark, setIsDark] = useState(false)
  
  const { 
    projects, 
    selectedProject, 
    setSelectedProject, 
    setProjects 
  } = useAppStore()

  const toggleTheme = () => {
    const newIsDark = !isDark
    setIsDark(newIsDark)
    if (newIsDark) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }

  const handleCreateProject = async () => {
    if (!newProjectName.trim()) return
    
    try {
      const project = await api.createProject(newProjectName)
      setProjects([project, ...projects])
      setSelectedProject(project)
      setNewProjectName('')
      setIsCreating(false)
    } catch (error) {
      console.error('Failed to create project:', error)
      alert('Failed to create project')
    }
  }

  return (
    <aside className="w-72 bg-surface border-r border-border flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-border flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Cpu size={20} className="text-secondary" />
          <h1 className="text-lg font-bold text-text">Research Assistant</h1>
        </div>
        <button
          onClick={toggleTheme}
          className="p-2 rounded-lg hover:bg-slate-200 dark:hover:bg-muted/20 transition-colors"
          title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {isDark ? <Sun size={18} /> : <Moon size={18} />}
        </button>
      </div>

      {/* Projects Section */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-muted uppercase tracking-wider">
            Projects
          </h2>
          <button
            onClick={() => setIsCreating(!isCreating)}
            className="p-1 hover:bg-slate-200 dark:hover:bg-muted/20 rounded transition-colors"
            title="Create new project"
          >
            <Plus size={18} />
          </button>
        </div>

        {/* Create Project Form */}
        {isCreating && (
          <div className="mb-4 p-3 bg-background rounded-lg border border-border shadow-sm">
            <input
              type="text"
              value={newProjectName}
              onChange={(e) => setNewProjectName(e.target.value)}
              placeholder="Project name"
              className="w-full px-3 py-2 bg-surface border border-border rounded text-sm focus:outline-none focus:border-secondary focus:ring-1 focus:ring-secondary"
              onKeyDown={(e) => e.key === 'Enter' && handleCreateProject()}
              autoFocus
            />
            <div className="flex gap-2 mt-2">
              <button
                onClick={handleCreateProject}
                className="flex-1 px-3 py-1.5 bg-secondary text-white text-sm rounded hover:bg-secondary-hover transition-colors"
              >
                Create
              </button>
              <button
                onClick={() => {
                  setIsCreating(false)
                  setNewProjectName('')
                }}
                className="flex-1 px-3 py-1.5 bg-slate-100 dark:bg-muted/20 text-sm rounded hover:bg-slate-200 dark:hover:bg-muted/30 transition-colors border border-border"
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        {/* Project List */}
        <div className="space-y-1">
          {projects.map((project) => (
            <button
              key={project.id}
              onClick={() => setSelectedProject(project)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left transition-colors ${
                selectedProject?.id === project.id
                  ? 'bg-secondary/10 text-secondary border border-secondary/20'
                  : 'hover:bg-slate-100 dark:hover:bg-muted/10 border border-transparent'
              }`}
            >
              <Folder size={18} className={
                selectedProject?.id === project.id ? 'text-secondary' : 'text-muted'
              } />
              <span className="flex-1 truncate text-sm">{project.name}</span>
            </button>
          ))}
          
          {projects.length === 0 && (
            <p className="text-sm text-muted text-center py-4">
              No projects yet
            </p>
          )}
        </div>
      </div>

      {/* Selected Project Info */}
      {selectedProject && (
        <div className="p-4 border-t border-border bg-surface/50">
          <p className="text-xs text-muted mb-1">Selected</p>
          <p className="text-sm font-medium truncate text-text">{selectedProject.name}</p>
          <p className="text-xs text-muted mt-1">ID: {selectedProject.id.slice(0, 8)}...</p>
        </div>
      )}
    </aside>
  )
}
