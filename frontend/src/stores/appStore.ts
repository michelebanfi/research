import { create } from 'zustand'

export interface Project {
  id: string
  name: string
  created_at: string
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  model?: string
}

export interface AgentEvent {
  type: string
  content: string
  metadata?: Record<string, any>
  timestamp_str?: string
}

export interface RetrievedChunk {
  id: string
  content: string
  similarity: number
  rerank_score?: number
  source: string
  file_path?: string
  metadata?: Record<string, any>
}

interface AppState {
  // Projects
  projects: Project[]
  selectedProject: Project | null
  setProjects: (projects: Project[]) => void
  setSelectedProject: (project: Project | null) => void
  
  // Chat
  chatHistory: ChatMessage[]
  addMessage: (message: ChatMessage) => void
  clearChat: () => void
  
  // Agent Events
  agentEvents: AgentEvent[]
  addAgentEvent: (event: AgentEvent) => void
  clearAgentEvents: () => void
  
  // Context
  retrievedChunks: RetrievedChunk[]
  matchedConcepts: string[]
  setRetrievedChunks: (chunks: RetrievedChunk[]) => void
  setMatchedConcepts: (concepts: string[]) => void
  
  // Settings
  doRerank: boolean
  topK: number
  reasoningMode: boolean
  setDoRerank: (value: boolean) => void
  setTopK: (value: number) => void
  setReasoningMode: (value: boolean) => void
  
  // UI State
  isChatting: boolean
  setIsChatting: (value: boolean) => void
  
  // Files
  files: any[]
  setFiles: (files: any[]) => void
}

export const useAppStore = create<AppState>((set) => ({
  // Projects
  projects: [],
  selectedProject: null,
  setProjects: (projects) => set({ projects }),
  setSelectedProject: (project) => set({ selectedProject: project }),
  
  // Chat
  chatHistory: [],
  addMessage: (message) => set((state) => ({ 
    chatHistory: [...state.chatHistory, message] 
  })),
  clearChat: () => set({ chatHistory: [] }),
  
  // Agent Events
  agentEvents: [],
  addAgentEvent: (event) => set((state) => ({ 
    agentEvents: [...state.agentEvents, event] 
  })),
  clearAgentEvents: () => set({ agentEvents: [] }),
  
  // Context
  retrievedChunks: [],
  matchedConcepts: [],
  setRetrievedChunks: (chunks) => set({ retrievedChunks: chunks }),
  setMatchedConcepts: (concepts) => set({ matchedConcepts: concepts }),
  
  // Settings
  doRerank: true,
  topK: 5,
  reasoningMode: false,
  setDoRerank: (value) => set({ doRerank: value }),
  setTopK: (value) => set({ topK: value }),
  setReasoningMode: (value) => set({ reasoningMode: value }),
  
  // UI State
  isChatting: false,
  setIsChatting: (value) => set({ isChatting: value }),
  
  // Files
  files: [],
  setFiles: (files) => set({ files }),
}))
