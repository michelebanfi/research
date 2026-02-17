import axios from 'axios'
import { Project, ChatMessage } from '../stores/appStore'

const API_BASE = '/api'

export const api = {
  // Projects
  async getProjects(): Promise<Project[]> {
    const response = await axios.get(`${API_BASE}/projects`)
    return response.data
  },

  async createProject(name: string): Promise<Project> {
    const response = await axios.post(`${API_BASE}/projects`, { name })
    return response.data
  },

  // Files
  async getProjectFiles(projectId: string): Promise<any[]> {
    const response = await axios.get(`${API_BASE}/projects/${projectId}/files`)
    return response.data
  },

  async deleteFile(fileId: string): Promise<void> {
    await axios.delete(`${API_BASE}/files/${fileId}`)
  },

  async getFileDetails(fileId: string): Promise<any> {
    const response = await axios.get(`${API_BASE}/files/${fileId}/details`)
    return response.data
  },

  async getUploadProgress(tempFileId: string): Promise<any> {
    const response = await axios.get(`${API_BASE}/upload/progress/${tempFileId}`)
    return response.data
  },

  async uploadFile(file: File, projectId: string): Promise<any> {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('project_id', projectId)

    const response = await axios.post(`${API_BASE}/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  },

  // Graph
  async getProjectGraph(projectId: string): Promise<{ nodes: any[], edges: any[] }> {
    const response = await axios.get(`${API_BASE}/projects/${projectId}/graph`)
    return response.data
  },
}

// WebSocket for chat - Singleton pattern
class ChatWebSocket {
  private ws: WebSocket | null = null
  private messageCallback: ((data: any) => void) | null = null
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectDelay = 1000
  private isIntentionallyClosed = false
  private url = ''

  connect(onMessage: (data: any) => void) {
    if (this.ws?.readyState === WebSocket.OPEN || this.ws?.readyState === WebSocket.CONNECTING) {
      console.log('WebSocket already connected or connecting')
      this.messageCallback = onMessage
      return
    }

    this.isIntentionallyClosed = false
    this.messageCallback = onMessage
    this.url = `ws://${window.location.host}/ws/chat`
    
    console.log('Connecting WebSocket...')
    this.ws = new WebSocket(this.url)

    this.ws.onopen = () => {
      console.log('WebSocket connected')
      this.reconnectAttempts = 0
    }

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        console.log('WebSocket message:', data.type)
        if (this.messageCallback) {
          this.messageCallback(data)
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error)
      }
    }

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error)
    }

    this.ws.onclose = (event) => {
      console.log('WebSocket disconnected:', event.code, event.reason)
      this.ws = null
      
      // Only reconnect if not intentionally closed
      if (!this.isIntentionallyClosed && this.reconnectAttempts < this.maxReconnectAttempts) {
        this.reconnectAttempts++
        const delay = this.reconnectDelay * this.reconnectAttempts
        console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`)
        setTimeout(() => this.connect(onMessage), delay)
      }
    }
  }

  send(message: {
    message: string
    project_id: string
    chat_history: ChatMessage[]
    do_rerank: boolean
    reasoning_mode: boolean
  }) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      console.log('Sending message via WebSocket')
      this.ws.send(JSON.stringify(message))
    } else {
      console.error('WebSocket not connected, state:', this.ws?.readyState)
      // Store message and retry? For now just error
      throw new Error('WebSocket not connected')
    }
  }

  disconnect() {
    console.log('Intentionally disconnecting WebSocket')
    this.isIntentionallyClosed = true
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN
  }
}

export const chatSocket = new ChatWebSocket()
