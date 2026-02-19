import { useState, useEffect } from 'react'
import { useAppStore } from './stores/appStore'
import { api } from './services/api'
import Sidebar from './components/Sidebar'
import ChatTab from './components/ChatTab'
import IngestTab from './components/IngestTab'
import GraphTab from './components/GraphTab'
import LangGraphTab from './components/LangGraphTab'
import PaperAnalysisTab from './components/PaperAnalysisTab'
import { MessageSquare, Upload, Share2, ArrowLeft, Radius, Microscope } from 'lucide-react'

type Tab = 'chat' | 'ingest' | 'graph' | 'langgraph' | 'analyze'

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('chat')
  const {
    selectedProject,
    setProjects,
    clearChat,
    clearAgentEvents,
    setCurrentChatId
  } = useAppStore()

  // Load projects on mount
  useEffect(() => {
    const loadProjects = async () => {
      try {
        const projects = await api.getProjects()
        setProjects(projects)
      } catch (error) {
        console.error('Failed to load projects:', error)
      }
    }
    loadProjects()
  }, [setProjects])

  // Load files when project changes
  useEffect(() => {
    const loadFiles = async () => {
      if (selectedProject) {
        try {
          const files = await api.getProjectFiles(selectedProject.id)
          useAppStore.getState().setFiles(files)
          // Clear chat history when project changes
          clearChat()
          clearAgentEvents()
          // Reset current chat ID
          setCurrentChatId(null)
        } catch (error) {
          console.error('Failed to load files:', error)
        }
      }
    }
    loadFiles()
  }, [selectedProject, clearChat, clearAgentEvents, setCurrentChatId])

  return (
    <div className="flex h-screen bg-background text-text">
      <Sidebar />

      <main className="flex-1 flex flex-col overflow-hidden">
        {selectedProject ? (
          <>
            {/* Tab Navigation */}
            <div className="flex items-center gap-1 px-4 py-2 bg-surface border-b border-border">
              <button
                onClick={() => setActiveTab('chat')}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === 'chat'
                    ? 'bg-secondary text-white shadow-sm'
                    : 'hover:bg-slate-100 dark:hover:bg-muted/20'
                  }`}
              >
                <MessageSquare size={16} />
                Chat
              </button>
              <button
                onClick={() => setActiveTab('ingest')}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === 'ingest'
                    ? 'bg-secondary text-white shadow-sm'
                    : 'hover:bg-slate-100 dark:hover:bg-muted/20'
                  }`}
              >
                <Upload size={16} />
                Ingest
              </button>
              <button
                onClick={() => setActiveTab('graph')}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === 'graph'
                    ? 'bg-secondary text-white shadow-sm'
                    : 'hover:bg-slate-100 dark:hover:bg-muted/20'
                  }`}
              >
                <Share2 size={16} />
                Graph
              </button>
              <button
                onClick={() => setActiveTab('langgraph')}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === 'langgraph'
                    ? 'bg-secondary text-white shadow-sm'
                    : 'hover:bg-slate-100 dark:hover:bg-muted/20'
                  }`}
              >
                <Radius size={16} />
                Architecture
              </button>
              <button
                onClick={() => setActiveTab('analyze')}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === 'analyze'
                    ? 'bg-secondary text-white shadow-sm'
                    : 'hover:bg-slate-100 dark:hover:bg-muted/20'
                  }`}
              >
                <Microscope size={16} />
                Analyze
              </button>
            </div>

            {/* Tab Content */}
            <div className="flex-1 overflow-hidden">
              {activeTab === 'chat' && <ChatTab />}
              {activeTab === 'ingest' && <IngestTab />}
              {activeTab === 'graph' && <GraphTab />}
              {activeTab === 'langgraph' && <LangGraphTab />}
              {activeTab === 'analyze' && <PaperAnalysisTab />}
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <ArrowLeft className="mx-auto mb-4 text-muted" size={48} />
              <p className="text-xl text-muted mb-4">
                Select or create a project to get started
              </p>
              <p className="text-sm text-muted/60">
                Your research assistant is ready to help
              </p>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
