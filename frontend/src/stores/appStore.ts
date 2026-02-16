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
  
  // Multiple chats support
  chats: Record<string, ChatMessage[]>  // chat_id -> messages
  currentChatId: string | null
  setCurrentChatId: (id: string | null) => void
  createNewChat: () => void
  switchChat: (id: string) => void
  
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
  addMessage: (message) => set((state) => {
    // Update both current chat history and stored chats
    const newHistory = [...state.chatHistory, message]
    const newChats = { ...state.chats }
    if (state.currentChatId) {
      newChats[state.currentChatId] = newHistory
    }
    return { chatHistory: newHistory, chats: newChats }
  }),
  clearChat: () => set({ chatHistory: [] }),
  
  // Multiple chats
  chats: {},
  currentChatId: null,
  setCurrentChatId: (id) => set({ currentChatId: id }),
  createNewChat: () => set((state) => {
    const newId = crypto.randomUUID()
    return {
      currentChatId: newId,
      chatHistory: [],
      chats: { ...state.chats, [newId]: [] },
      agentEvents: [],
      retrievedChunks: [],
      matchedConcepts: []
    }
  }),
  switchChat: (id) => set((state) => ({
    currentChatId: id,
    chatHistory: state.chats[id] || [],
    agentEvents: [],
    retrievedChunks: [],
    matchedConcepts: []
  })),
  
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
