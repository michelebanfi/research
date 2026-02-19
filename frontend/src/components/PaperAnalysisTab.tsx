import { useState, useEffect, useRef } from 'react'
import { useAppStore, AnalysisEvent } from '../stores/appStore'
import { analysisSocket } from '../services/api'
import RunCodeBlock from './RunCodeBlock'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import {
    Microscope,
    FileText,
    Play,
    Download,
    Loader2,
    CheckCircle2,
    Clock,
    AlertCircle,
    ChevronDown,
    BookOpen,
    Layers,
    Sparkles,
    Wifi,
    WifiOff,
} from 'lucide-react'

type SectionStatus = 'pending' | 'analyzing' | 'done' | 'error'

interface SectionEntry {
    title: string
    status: SectionStatus
    preview: string
    index: number
}

export default function PaperAnalysisTab() {
    const {
        selectedProject,
        files,
        analysisStatus,
        analysisMarkdown,
        selectedAnalysisFileId,
        setAnalysisStatus,
        addAnalysisEvent,
        setAnalysisMarkdown,
        setSelectedAnalysisFileId,
        resetAnalysis,
    } = useAppStore()

    const [sections, setSections] = useState<SectionEntry[]>([])
    const [isConnected, setIsConnected] = useState(false)
    const [errorMessage, setErrorMessage] = useState<string | null>(null)
    const [totalSections, setTotalSections] = useState(0)
    const outputEndRef = useRef<HTMLDivElement>(null)
    const scrollContainerRef = useRef<HTMLDivElement>(null)

    // Keep track of connection status
    useEffect(() => {
        const interval = setInterval(() => {
            setIsConnected(analysisSocket.isConnected())
        }, 1000)
        return () => clearInterval(interval)
    }, [])

    // Auto-scroll output as content comes in — scroll the container directly
    // to avoid scrollIntoView accidentally scrolling the page.
    useEffect(() => {
        if (analysisStatus === 'running' && scrollContainerRef.current) {
            const el = scrollContainerRef.current
            el.scrollTop = el.scrollHeight
        }
    }, [analysisMarkdown, analysisStatus])

    // Connect WebSocket on mount
    useEffect(() => {
        analysisSocket.connect((data) => {
            if (data.type === 'analysis_event') {
                const event: AnalysisEvent = data.event
                addAnalysisEvent(event)

                if (event.type === 'start') {
                    setTotalSections(event.total_sections)
                    // Seed section entries as pending
                    setSections(
                        Array.from({ length: event.total_sections }, (_, i) => ({
                            title: `Section ${i + 1}`,
                            status: 'pending' as SectionStatus,
                            preview: '',
                            index: i,
                        }))
                    )
                    setAnalysisStatus('running')
                } else if (event.type === 'section') {
                    setSections((prev) => {
                        const next = [...prev]
                        const idx = event.section_index
                        // Mark previous ones done, current done too
                        for (let i = 0; i <= idx && i < next.length; i++) {
                            if (next[i].status !== 'done') {
                                next[i] = {
                                    ...next[i],
                                    title: i === idx ? event.section_title : next[i].title,
                                    status: 'done',
                                    preview: i === idx ? event.content.slice(0, 200) : next[i].preview,
                                }
                            }
                        }
                        // Update title of next pending section if we know it
                        if (idx + 1 < next.length && next[idx + 1].status === 'pending') {
                            next[idx + 1] = { ...next[idx + 1], status: 'analyzing' }
                        }
                        return next
                    })
                } else if (event.type === 'complete') {
                    setAnalysisMarkdown(event.content)
                    setAnalysisStatus('complete')
                    setSections((prev) => prev.map((s) => ({ ...s, status: 'done' as SectionStatus })))
                } else if (event.type === 'error') {
                    setAnalysisStatus('error')
                    setErrorMessage(event.content)
                }
            } else if (data.type === 'status') {
                // status update, no-op for now
            } else if (data.type === 'result') {
                if (data.content?.success) {
                    setAnalysisMarkdown(data.content.markdown)
                }
                setAnalysisStatus('complete')
            } else if (data.type === 'error') {
                setAnalysisStatus('error')
                setErrorMessage(data.content)
            }
        })

        return () => {
            analysisSocket.disconnect()
        }
    }, [addAnalysisEvent, setAnalysisStatus, setAnalysisMarkdown])

    const handleAnalyze = () => {
        if (!selectedAnalysisFileId || !selectedProject) return
        if (analysisStatus === 'running') return

        // Reset previous state
        resetAnalysis()
        setSections([])
        setErrorMessage(null)
        setTotalSections(0)
        setAnalysisStatus('connecting')

        try {
            // The real message handler is already set up by the useEffect on mount.
            // Do NOT call analysisSocket.connect() again here — that would overwrite
            // the callback and swallow all incoming events.
            setTimeout(() => {
                try {
                    analysisSocket.send({
                        file_id: selectedAnalysisFileId,
                        project_id: selectedProject.id,
                    })
                } catch (e) {
                    setAnalysisStatus('error')
                    setErrorMessage('Could not connect to the server. Is the backend running?')
                }
            }, 300)
        } catch (e) {
            setAnalysisStatus('error')
            setErrorMessage(`Failed to start analysis: ${e}`)
        }
    }

    const handleDownload = () => {
        if (!analysisMarkdown) return
        const selectedFile = files.find((f) => f.id === selectedAnalysisFileId)
        const baseName = selectedFile?.name?.replace(/\.[^.]+$/, '') ?? 'analysis'
        const fileName = `${baseName}_analysis.md`
        const blob = new Blob([analysisMarkdown], { type: 'text/markdown' })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = fileName
        a.click()
        URL.revokeObjectURL(url)
    }

    const isRunning = analysisStatus === 'running' || analysisStatus === 'connecting'
    const selectedFileData = files.find((f) => f.id === selectedAnalysisFileId)

    return (
        <div className="flex h-full overflow-hidden">
            {/* ── Main Output Area ─────────────────────────────────────── */}
            <div className="flex-1 flex flex-col min-w-0 min-h-0">
                {/* Header */}
                <div className="flex items-center justify-between px-4 py-2 border-b border-border">
                    <div className="flex items-center gap-2">
                        <Microscope size={20} className="text-secondary" />
                        <h2 className="text-lg font-semibold">Paper Analysis</h2>
                        <div className={`flex items-center gap-1 px-2 py-1 rounded text-xs ${isConnected ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-400' : 'bg-slate-100 text-slate-500 dark:bg-muted/20 dark:text-muted'}`}>
                            {isConnected ? <Wifi size={12} /> : <WifiOff size={12} />}
                            {isConnected ? 'Ready' : 'Disconnected'}
                        </div>
                    </div>

                    {analysisStatus === 'complete' && (
                        <button
                            onClick={handleDownload}
                            className="flex items-center gap-2 px-4 py-2 bg-secondary text-white rounded-lg hover:bg-secondary-hover transition-all text-sm font-medium shadow-sm"
                        >
                            <Download size={16} />
                            Download .md
                        </button>
                    )}
                </div>

                {/* File Picker + Analyze Button */}
                <div className="px-4 py-3 border-b border-border bg-surface/50">
                    <div className="flex items-center gap-3">
                        <div className="relative flex-1">
                            <FileText size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted pointer-events-none" />
                            <select
                                value={selectedAnalysisFileId ?? ''}
                                onChange={(e) => setSelectedAnalysisFileId(e.target.value || null)}
                                disabled={isRunning}
                                className="w-full pl-9 pr-8 py-2.5 bg-background border border-border rounded-lg text-sm focus:outline-none focus:border-secondary focus:ring-2 focus:ring-secondary/20 disabled:opacity-50 appearance-none cursor-pointer"
                            >
                                <option value="">Select an ingested paper...</option>
                                {files.map((f) => (
                                    <option key={f.id} value={f.id}>{f.name}</option>
                                ))}
                            </select>
                            <ChevronDown size={14} className="absolute right-3 top-1/2 -translate-y-1/2 text-muted pointer-events-none" />
                        </div>

                        <button
                            onClick={handleAnalyze}
                            disabled={!selectedAnalysisFileId || isRunning}
                            className="flex items-center gap-2 px-5 py-2.5 bg-secondary text-white rounded-lg hover:bg-secondary-hover disabled:opacity-50 disabled:cursor-not-allowed transition-all text-sm font-medium shadow-sm hover:shadow-md whitespace-nowrap"
                        >
                            {isRunning ? (
                                <><Loader2 size={16} className="animate-spin" /> Analyzing...</>
                            ) : (
                                <><Play size={16} /> Analyze Paper</>
                            )}
                        </button>
                    </div>

                    {/* File info row */}
                    {selectedFileData && (
                        <p className="text-xs text-muted mt-2 flex items-center gap-1">
                            <BookOpen size={11} />
                            {selectedFileData.name}
                            {totalSections > 0 && (
                                <span className="ml-1 text-secondary font-medium">· {totalSections} sections to analyze</span>
                            )}
                        </p>
                    )}
                </div>

                {/* Output Area */}
                <div ref={scrollContainerRef} className="flex-1 min-h-0 overflow-y-auto p-4">
                    {analysisStatus === 'idle' && (
                        <div className="flex flex-col items-center justify-center h-full text-center gap-4 text-muted">
                            <Microscope size={56} className="opacity-20" />
                            <div>
                                <p className="text-lg font-medium mb-1">Paper Analysis</p>
                                <p className="text-sm max-w-sm">
                                    Select an ingested paper and click <strong>Analyze Paper</strong> to generate a
                                    comprehensive section-by-section analysis with plain-English explanations,
                                    math breakdowns, and visualization code.
                                </p>
                            </div>
                        </div>
                    )}

                    {(analysisStatus === 'connecting') && (
                        <div className="flex flex-col items-center justify-center h-full gap-4 text-muted">
                            <Loader2 size={32} className="animate-spin text-secondary" />
                            <p className="text-sm">Connecting to analysis server...</p>
                        </div>
                    )}

                    {analysisStatus === 'error' && (
                        <div className="flex flex-col items-center justify-center h-full gap-4">
                            <AlertCircle size={40} className="text-red-500" />
                            <div className="text-center">
                                <p className="font-medium text-red-600 dark:text-red-400 mb-2">Analysis Failed</p>
                                <p className="text-sm text-muted max-w-sm">{errorMessage}</p>
                                <button
                                    onClick={() => { resetAnalysis(); setSections([]) }}
                                    className="mt-4 px-4 py-2 bg-secondary text-white rounded-lg text-sm hover:bg-secondary-hover"
                                >
                                    Try Again
                                </button>
                            </div>
                        </div>
                    )}

                    {(analysisStatus === 'running' || analysisStatus === 'complete') && analysisMarkdown && (
                        <div className="max-w-4xl mx-auto">
                            <div className="markdown-content prose prose-sm dark:prose-invert max-w-none">
                                <ReactMarkdown
                                    remarkPlugins={[remarkMath]}
                                    rehypePlugins={[rehypeKatex]}
                                    components={{
                                        code({ node, inline, className, children, ...props }: any) {
                                            const language = (className ?? '').replace('language-', '')
                                            const codeText = String(children).replace(/\n$/, '')
                                            if (!inline) {
                                                return (
                                                    <RunCodeBlock
                                                        code={codeText}
                                                        language={language}
                                                    />
                                                )
                                            }
                                            return (
                                                <code className="bg-slate-100 dark:bg-slate-800 px-1.5 py-0.5 rounded text-sm" {...props}>
                                                    {children}
                                                </code>
                                            )
                                        },
                                        h1({ children }) {
                                            return <h1 className="text-2xl font-bold mt-6 mb-3 pb-2 border-b border-border">{children}</h1>
                                        },
                                        h2({ children }) {
                                            return <h2 className="text-xl font-semibold mt-5 mb-2 text-secondary">{children}</h2>
                                        },
                                        h3({ children }) {
                                            return <h3 className="text-base font-semibold mt-4 mb-1">{children}</h3>
                                        },
                                        ul({ children }) {
                                            return <ul className="list-disc pl-5 my-2 space-y-1">{children}</ul>
                                        },
                                        ol({ children }) {
                                            return <ol className="list-decimal pl-5 my-2 space-y-1">{children}</ol>
                                        },
                                        blockquote({ children }) {
                                            return (
                                                <blockquote className="border-l-4 border-secondary/40 pl-4 my-3 italic text-muted">
                                                    {children}
                                                </blockquote>
                                            )
                                        },
                                        hr() {
                                            return <hr className="my-6 border-border" />
                                        },
                                    }}
                                >
                                    {analysisMarkdown}
                                </ReactMarkdown>
                                <div ref={outputEndRef} />
                            </div>
                        </div>
                    )}

                    {analysisStatus === 'running' && !analysisMarkdown && (
                        <div className="flex flex-col items-center justify-center h-full gap-3 text-muted">
                            <Loader2 size={28} className="animate-spin text-secondary" />
                            <p className="text-sm">Analyzing sections — results will appear as each section completes...</p>
                        </div>
                    )}
                </div>
            </div>

            {/* ── Right Panel — Live Progress ───────────────────────────── */}
            <div className="w-80 border-l border-border flex flex-col bg-surface/50 shrink-0">
                <div className="px-4 py-2 border-b border-border font-semibold text-sm flex items-center gap-2">
                    <Layers size={16} />
                    Section Progress
                    {totalSections > 0 && (
                        <span className="ml-auto text-xs text-muted">
                            {sections.filter(s => s.status === 'done').length}/{totalSections}
                        </span>
                    )}
                </div>

                <div className="flex-1 overflow-y-auto p-3 space-y-2">
                    {sections.length === 0 && (
                        <p className="text-sm text-muted text-center py-8">
                            {analysisStatus === 'idle'
                                ? 'Section progress will appear here during analysis.'
                                : 'Waiting for sections...'}
                        </p>
                    )}

                    {sections.map((section, idx) => (
                        <SectionProgressCard key={idx} section={section} />
                    ))}
                </div>

                {/* Summary footer */}
                {analysisStatus === 'complete' && (
                    <div className="p-3 border-t border-border bg-emerald-50 dark:bg-emerald-500/10">
                        <div className="flex items-center gap-2 text-emerald-700 dark:text-emerald-400 text-sm font-medium">
                            <Sparkles size={16} />
                            Analysis complete!
                        </div>
                        <p className="text-xs text-muted mt-1">
                            {sections.length} sections analyzed · {Math.round(analysisMarkdown.length / 1000)}k chars
                        </p>
                    </div>
                )}
            </div>
        </div>
    )
}

// ---------------------------------------------------------------------------
// Section progress card
// ---------------------------------------------------------------------------

function SectionProgressCard({ section }: { section: SectionEntry }) {
    const { status, title } = section

    const icon = () => {
        switch (status) {
            case 'done': return <CheckCircle2 size={14} className="text-emerald-500 shrink-0" />
            case 'analyzing': return <Loader2 size={14} className="text-secondary animate-spin shrink-0" />
            case 'error': return <AlertCircle size={14} className="text-red-500 shrink-0" />
            default: return <Clock size={14} className="text-muted shrink-0" />
        }
    }

    const bg = () => {
        switch (status) {
            case 'done': return 'bg-emerald-50 border-emerald-200 dark:bg-emerald-500/10 dark:border-emerald-500/30'
            case 'analyzing': return 'bg-blue-50 border-blue-200 dark:bg-blue-500/10 dark:border-blue-500/30 animate-pulse'
            case 'error': return 'bg-red-50 border-red-200 dark:bg-red-500/10 dark:border-red-500/30'
            default: return 'bg-slate-50 border-slate-200 dark:bg-muted/10 dark:border-muted/20'
        }
    }

    return (
        <div className={`flex items-start gap-2 p-2.5 rounded-lg border text-sm ${bg()}`}>
            {icon()}
            <div className="min-w-0">
                <p className={`font-medium truncate text-xs ${status === 'pending' ? 'text-muted' : ''}`}>
                    {section.index + 1}. {title}
                </p>
                {status === 'analyzing' && (
                    <p className="text-xs text-muted mt-0.5">Analyzing...</p>
                )}
                {status === 'done' && section.preview && (
                    <p className="text-xs text-muted mt-0.5 line-clamp-2 leading-relaxed">
                        {section.preview.replace(/^#+\s+/gm, '').replace(/\*\*/g, '').slice(0, 120)}...
                    </p>
                )}
            </div>
        </div>
    )
}
