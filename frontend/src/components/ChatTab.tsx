import { useState, useRef, useEffect } from 'react'
import { useAppStore, AgentEvent } from '../stores/appStore'
import { chatSocket } from '../services/api'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { Logo } from './Logo'
import { 
  Send, 
  Settings, 
  Brain, 
  Loader2, 
  Wifi, 
  WifiOff, 
  Activity,
  BookOpen,
  Lightbulb,
  Wrench,
  CheckCircle2,
  Search,
  Sparkles,
  AlertCircle,
  FileText,
  Globe,
  Network,
  Plus
} from 'lucide-react'

export default function ChatTab() {
  const [input, setInput] = useState('')
  const [currentResponse, setCurrentResponse] = useState('')
  const [showSettings, setShowSettings] = useState(false)
  const [isConnected, setIsConnected] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  
  const {
    selectedProject,
    chatHistory,
    addMessage,
    agentEvents,
    addAgentEvent,
    clearAgentEvents,
    doRerank,
    setDoRerank,
    topK,
    setTopK,
    reasoningMode,
    setReasoningMode,
    isChatting,
    setIsChatting,
    retrievedChunks,
    matchedConcepts,
    setRetrievedChunks,
    setMatchedConcepts,
    createNewChat,
  } = useAppStore()

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chatHistory, currentResponse])

  // Connect WebSocket on mount
  useEffect(() => {
    const checkConnection = () => {
      setIsConnected(chatSocket.isConnected())
    }
    
    // Check connection status periodically
    const interval = setInterval(checkConnection, 1000)
    
    chatSocket.connect((data) => {
      if (data.type === 'event') {
        // Filter out token events - they're just streamed answer fragments
        if (data.event.type !== 'token') {
          addAgentEvent(data.event)
        }
      } else if (data.type === 'status') {
        // Status update
      } else if (data.type === 'result') {
        setIsChatting(false)
        const result = data.content
        if (result.success) {
          addMessage({
            role: 'assistant',
            content: result.answer,
            model: result.model_name,
          })
          if (result.retrieved_chunks) {
            setRetrievedChunks(result.retrieved_chunks)
          }
          if (result.matched_concepts) {
            setMatchedConcepts(result.matched_concepts)
          }
        }
        setCurrentResponse('')
      } else if (data.type === 'error') {
        setIsChatting(false)
        addMessage({
          role: 'assistant',
          content: `Error: ${data.content}`,
        })
        setCurrentResponse('')
      }
    })

    return () => {
      clearInterval(interval)
      chatSocket.disconnect()
    }
  }, [addMessage, addAgentEvent, setIsChatting, setRetrievedChunks, setMatchedConcepts])

  const handleSend = () => {
    if (!input.trim() || !selectedProject || isChatting) return

    if (!isConnected) {
      addMessage({
        role: 'assistant',
        content: 'Not connected to server. Please wait for connection or refresh the page.',
      })
      return
    }

    // Create a new chat if none exists
    const { currentChatId, createNewChat } = useAppStore.getState()
    if (!currentChatId) {
      createNewChat()
    }

    // Add user message
    addMessage({ role: 'user', content: input })
    
    // Clear previous events
    clearAgentEvents()
    setCurrentResponse('')
    setIsChatting(true)

    // Send via WebSocket
    try {
      chatSocket.send({
        message: input,
        project_id: selectedProject.id,
        chat_history: chatHistory.slice(0, -1), // Exclude the message we just added
        do_rerank: doRerank,
        reasoning_mode: reasoningMode,
      })
    } catch (error) {
      setIsChatting(false)
      addMessage({
        role: 'assistant',
        content: 'Failed to send message. Please try again.',
      })
    }

    setInput('')
  }

  return (
    <div className="flex h-full">
      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Chat Header with Settings and New Chat Button */}
        <div className="flex items-center justify-between px-4 py-2 border-b border-border">
          <div className="flex items-center gap-2">
            <h2 className="text-lg font-semibold">Chat</h2>
            <div className={`flex items-center gap-1 px-2 py-1 rounded text-xs ${isConnected ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-400' : 'bg-red-100 text-red-700 dark:bg-red-500/20 dark:text-red-400'}`}>
              {isConnected ? <Wifi size={12} /> : <WifiOff size={12} />}
              {isConnected ? 'Connected' : 'Disconnected'}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={createNewChat}
              className="flex items-center gap-2 px-3 py-1.5 text-sm bg-secondary text-white rounded-lg hover:bg-secondary-hover transition-colors"
            >
              <Plus size={16} />
              New Chat
            </button>
            <button
              onClick={() => setShowSettings(!showSettings)}
              className={`p-2 rounded-lg transition-colors ${
                showSettings 
                  ? 'bg-slate-200 text-secondary dark:bg-secondary/20 dark:text-secondary' 
                  : 'hover:bg-slate-100 dark:hover:bg-muted/20'
              }`}
            >
              <Settings size={20} />
            </button>
          </div>
        </div>

        {/* Settings Panel */}
        {showSettings && (
          <div className="px-4 py-3 bg-surface border-b border-border space-y-3">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={doRerank}
                onChange={(e) => setDoRerank(e.target.checked)}
                className="rounded border-muted/30"
              />
              <span className="text-sm">Enable Re-ranking</span>
            </label>
            
            <div className="flex items-center gap-3">
              <span className="text-sm">Context chunks:</span>
              <input
                type="range"
                min={3}
                max={10}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="flex-1"
              />
              <span className="text-sm w-8">{topK}</span>
            </div>
            
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={reasoningMode}
                onChange={(e) => setReasoningMode(e.target.checked)}
                className="rounded border-muted/30"
              />
              <span className="text-sm flex items-center gap-1">
                <Brain size={14} />
                Reasoning Mode (Plan & Code)
              </span>
            </label>
          </div>
        )}

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {/* Show Logo when no chat history */}
          {chatHistory.length === 0 && !isChatting && (
            <div className="flex-1 flex items-center justify-center h-full">
              <Logo />
            </div>
          )}
          
          {chatHistory.map((msg, idx) => (
            <div
              key={idx}
              className={`flex ${
                msg.role === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              <div
                className={`max-w-[80%] rounded-lg px-4 py-3 shadow-sm ${
                  msg.role === 'user'
                    ? 'bg-secondary text-white'
                    : 'bg-white border border-border dark:bg-surface'
                }`}
              >
                {msg.role === 'user' ? (
                  <div className="whitespace-pre-wrap text-sm">{msg.content}</div>
                ) : (
                  <MarkdownMessage content={msg.content} />
                )}
                {msg.model && (
                  <div className="text-xs opacity-60 mt-2">
                    Generated by: {msg.model}
                  </div>
                )}
              </div>
            </div>
          ))}
          
          {isChatting && (
            <div className="flex justify-start">
              <div className="bg-white border border-border dark:bg-surface rounded-lg px-4 py-3 flex items-center gap-2 shadow-sm">
                <Loader2 size={16} className="animate-spin text-secondary" />
                <span className="text-sm">Thinking...</span>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="p-4 border-t border-border">
          <div className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
              placeholder="Ask a question about your knowledge base..."
              disabled={isChatting}
              className="flex-1 px-4 py-2.5 bg-background border border-border rounded-lg focus:outline-none focus:border-secondary focus:ring-2 focus:ring-secondary/20 disabled:opacity-50 shadow-sm"
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || isChatting}
              className="px-4 py-2.5 bg-secondary text-white rounded-lg hover:bg-secondary-hover disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-sm hover:shadow-md"
            >
              <Send size={20} />
            </button>
          </div>
        </div>
      </div>

      {/* Right Panel - Process Monitor & Context */}
      <div className="w-96 border-l border-border flex flex-col bg-surface/50">
        {/* Process Monitor */}
        <div className="flex-1 flex flex-col min-h-0">
          <div className="px-4 py-2 border-b border-border font-semibold text-sm flex items-center gap-2">
            <Activity size={16} />
            Live Process
          </div>
          <div className="flex-1 overflow-y-auto p-4 space-y-2">
            {agentEvents.length === 0 ? (
              <p className="text-sm text-muted text-center py-4">
                No active process. Start a chat to see the agent's reasoning.
              </p>
            ) : (
              agentEvents.map((event, idx) => (
                <AgentEventCard key={idx} event={event} />
              ))
            )}
          </div>
        </div>

        {/* Context Panel */}
        <div className="flex-1 flex flex-col min-h-0 border-t border-border">
          <div className="px-4 py-2 border-b border-border font-semibold text-sm flex items-center gap-2">
            <BookOpen size={16} />
            Context
          </div>
          <div className="flex-1 overflow-y-auto p-4">
            {matchedConcepts.length > 0 && (
              <div className="mb-4">
                <p className="text-xs text-muted mb-2">Matched Concepts:</p>
                <div className="flex flex-wrap gap-1">
                  {matchedConcepts.slice(0, 8).map((concept, idx) => (
                    <span
                      key={idx}
                      className="px-2 py-0.5 bg-slate-200 text-slate-700 dark:bg-secondary/20 dark:text-secondary text-xs rounded font-medium"
                    >
                      {concept}
                    </span>
                  ))}
                </div>
              </div>
            )}
            
            {retrievedChunks.length === 0 ? (
              <p className="text-sm text-muted text-center py-4">
                No context retrieved yet.
              </p>
            ) : (
              <div className="space-y-3">
                {retrievedChunks.map((chunk, idx) => (
                  <ContextChunk key={idx} chunk={chunk} index={idx} />
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function AgentEventCard({ event }: { event: AgentEvent }) {
  const getIcon = () => {
    switch (event.type) {
      case 'thought': return <Lightbulb size={16} />
      case 'tool': return <Wrench size={16} />
      case 'tool_result': return <CheckCircle2 size={16} />
      case 'search': return <Search size={16} />
      case 'result': return <Sparkles size={16} />
      case 'error': return <AlertCircle size={16} />
      default: return <FileText size={16} />
    }
  }

  const getIconColor = () => {
    switch (event.type) {
      case 'thought': return 'text-blue-600 dark:text-blue-400'
      case 'tool': return 'text-amber-600 dark:text-yellow-400'
      case 'tool_result': return 'text-emerald-600 dark:text-green-400'
      case 'search': return 'text-violet-600 dark:text-purple-400'
      case 'result': return 'text-emerald-600 dark:text-green-400'
      case 'error': return 'text-red-600 dark:text-red-400'
      default: return 'text-slate-600 dark:text-muted'
    }
  }

  const getBgColor = () => {
    switch (event.type) {
      case 'thought': return 'bg-blue-50 border-blue-200 dark:bg-blue-500/10 dark:border-blue-500/30'
      case 'tool': return 'bg-amber-50 border-amber-200 dark:bg-yellow-500/10 dark:border-yellow-500/30'
      case 'tool_result': return 'bg-emerald-50 border-emerald-200 dark:bg-green-500/10 dark:border-green-500/30'
      case 'search': return 'bg-violet-50 border-violet-200 dark:bg-purple-500/10 dark:border-purple-500/30'
      case 'result': return 'bg-emerald-100 border-emerald-300 dark:bg-green-500/20 dark:border-green-500/50'
      case 'error': return 'bg-red-50 border-red-200 dark:bg-red-500/10 dark:border-red-500/30'
      default: return 'bg-slate-50 border-slate-200 dark:bg-muted/10 dark:border-muted/30'
    }
  }

  return (
    <div className={`p-3 rounded-lg border text-sm ${getBgColor()}`}>
      <div className="flex items-start gap-2">
        <span className={getIconColor()}>{getIcon()}</span>
        <div className="flex-1 min-w-0">
          <p className="font-medium capitalize">{event.type}</p>
          <p className="text-muted mt-1">{event.content}</p>
              {event.metadata && Object.keys(event.metadata).length > 0 && (
            <details className="mt-2">
              <summary className="text-xs text-muted cursor-pointer">Details</summary>
              <pre className="mt-1 text-xs bg-slate-100 dark:bg-black/20 p-2 rounded overflow-x-auto">
                {JSON.stringify(event.metadata, null, 2)}
              </pre>
            </details>
          )}
        </div>
      </div>
    </div>
  )
}

// Markdown component with math support for assistant messages
function MarkdownMessage({ content }: { content: string }) {
  return (
    <div className="markdown-content prose prose-sm dark:prose-invert max-w-none">
      <ReactMarkdown
        remarkPlugins={[remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{
          // Style code blocks
          code({ node, inline, className, children, ...props }: any) {
            return !inline ? (
              <pre className="bg-slate-100 dark:bg-slate-800 p-3 rounded-lg overflow-x-auto my-2">
                <code className={className} {...props}>
                  {children}
                </code>
              </pre>
            ) : (
              <code className="bg-slate-100 dark:bg-slate-800 px-1.5 py-0.5 rounded text-sm" {...props}>
                {children}
              </code>
            )
          },
          // Style lists
          ul({ children }) {
            return <ul className="list-disc pl-5 my-2">{children}</ul>
          },
          ol({ children }) {
            return <ol className="list-decimal pl-5 my-2">{children}</ol>
          },
          // Style links
          a({ children, href }) {
            return (
              <a href={href} className="text-blue-600 dark:text-blue-400 hover:underline" target="_blank" rel="noopener noreferrer">
                {children}
              </a>
            )
          },
          // Style blockquotes
          blockquote({ children }) {
            return (
              <blockquote className="border-l-4 border-slate-300 dark:border-slate-600 pl-4 my-2 italic">
                {children}
              </blockquote>
            )
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}

function ContextChunk({ chunk, index }: { chunk: any; index: number }) {
  const [expanded, setExpanded] = useState(index === 0)
  
  const getSourceIcon = () => {
    switch (chunk.source) {
      case 'vector': return <Search size={14} />
      case 'web': return <Globe size={14} />
      case 'graph': return <Network size={14} />
      default: return <FileText size={14} />
    }
  }

  const getSourceLabel = () => {
    switch (chunk.source) {
      case 'vector': return 'Vector Search'
      case 'web': return 'Web Search'
      case 'graph': return 'Graph Retrieval'
      default: return 'Unknown'
    }
  }

  return (
    <div className="bg-background rounded-lg border border-border overflow-hidden shadow-sm">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-3 py-2 flex items-center justify-between text-left hover:bg-slate-50 dark:hover:bg-muted/10 transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="text-muted">{getSourceIcon()}</span>
          <span className="text-sm font-medium">Source {index + 1}</span>
        </div>
        <span className="text-xs text-muted">{getSourceLabel()}</span>
      </button>
      
      {expanded && (
        <div className="px-3 pb-3">
          <div className="text-xs text-muted mb-2">
            Similarity: {chunk.similarity?.toFixed(3) || 'N/A'}
            {chunk.rerank_score && ` → Re-ranked: ${chunk.rerank_score.toFixed(3)}`}
          </div>
          <p className="text-sm text-muted line-clamp-4">
            {chunk.content?.slice(0, 500)}
            {chunk.content?.length > 500 && '...'}
          </p>
        </div>
      )}
    </div>
  )
}
