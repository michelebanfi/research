/**
 * RunCodeBlock — renders a code block with syntax highlighting and an
 * inline "Run" button for Python snippets.
 *
 * Uses react-syntax-highlighter with a light theme so code is readable
 * against the light UI background.  Unknown / generic languages (e.g.
 * "code") fall back to plain-text rendering.
 */

import { useState } from 'react'
import { api } from '../services/api'
import { Light as SyntaxHighlighter } from 'react-syntax-highlighter'
import python from 'react-syntax-highlighter/dist/esm/languages/hljs/python'
import javascript from 'react-syntax-highlighter/dist/esm/languages/hljs/javascript'
import bash from 'react-syntax-highlighter/dist/esm/languages/hljs/bash'
import json from 'react-syntax-highlighter/dist/esm/languages/hljs/json'
import latex from 'react-syntax-highlighter/dist/esm/languages/hljs/latex'
import { githubGist } from 'react-syntax-highlighter/dist/esm/styles/hljs'
import {
    Play,
    Loader2,
    TerminalSquare,
    CheckCircle2,
    AlertCircle,
    RefreshCw,
    Copy,
    Check,
} from 'lucide-react'

// Register only the languages we actually need (keeps bundle small)
SyntaxHighlighter.registerLanguage('python', python)
SyntaxHighlighter.registerLanguage('py', python)
SyntaxHighlighter.registerLanguage('javascript', javascript)
SyntaxHighlighter.registerLanguage('js', javascript)
SyntaxHighlighter.registerLanguage('bash', bash)
SyntaxHighlighter.registerLanguage('shell', bash)
SyntaxHighlighter.registerLanguage('sh', bash)
SyntaxHighlighter.registerLanguage('json', json)
SyntaxHighlighter.registerLanguage('latex', latex)
SyntaxHighlighter.registerLanguage('tex', latex)

interface Props {
    code: string
    language: string
}

/** Normalise unknown / empty language tags to a safe fallback. */
function normaliseLang(lang: string): string {
    const l = (lang || '').toLowerCase().trim()
    const known = [
        'python', 'py', 'javascript', 'js', 'bash', 'shell', 'sh',
        'json', 'latex', 'tex',
    ]
    if (known.includes(l)) return l
    return 'text'
}

export default function RunCodeBlock({ code, language }: Props) {
    const [status, setStatus] = useState<'idle' | 'running' | 'success' | 'error'>('idle')
    const [output, setOutput] = useState<string>('')
    const [images, setImages] = useState<string[]>([])
    const [copied, setCopied] = useState(false)

    const lang = normaliseLang(language)
    const isPython = lang === 'python' || lang === 'py'

    const handleRun = async () => {
        setStatus('running')
        setOutput('')
        setImages([])
        try {
            const result = await api.runCode(code, 20)
            setStatus(result.success ? 'success' : 'error')
            setOutput(result.output)
            setImages(result.image_b64 ?? [])
        } catch (e: any) {
            setStatus('error')
            setOutput(e?.message ?? String(e))
        }
    }

    const handleReset = () => {
        setStatus('idle')
        setOutput('')
        setImages([])
    }

    const handleCopy = async () => {
        await navigator.clipboard.writeText(code)
        setCopied(true)
        setTimeout(() => setCopied(false), 2000)
    }

    return (
        <div className="code-block-wrapper my-4 rounded-xl border border-slate-200 overflow-hidden shadow-sm">
            {/* ── Header bar ─────────────────────────────────────────── */}
            <div className="flex items-center justify-between px-4 py-2 bg-slate-100 border-b border-slate-200">
                <span className="font-mono text-xs font-medium text-slate-500 tracking-wide uppercase">
                    {language || 'code'}
                </span>

                <div className="flex items-center gap-1.5">
                    {/* Copy button */}
                    <button
                        onClick={handleCopy}
                        className="flex items-center gap-1 px-2 py-1 rounded-md text-xs text-slate-500 hover:text-slate-700 hover:bg-slate-200/70 transition-colors"
                        title="Copy code"
                    >
                        {copied ? <Check size={12} className="text-emerald-600" /> : <Copy size={12} />}
                        {copied ? 'Copied' : 'Copy'}
                    </button>

                    {/* Reset button */}
                    {status !== 'idle' && (
                        <button
                            onClick={handleReset}
                            className="flex items-center gap-1 px-2 py-1 rounded-md text-xs text-slate-500 hover:text-slate-700 hover:bg-slate-200/70 transition-colors"
                            title="Reset output"
                        >
                            <RefreshCw size={11} />
                            Reset
                        </button>
                    )}

                    {/* Run button (Python only) */}
                    {isPython && (
                        <button
                            onClick={handleRun}
                            disabled={status === 'running'}
                            className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-60 text-white rounded-md text-xs font-semibold transition-colors shadow-sm"
                        >
                            {status === 'running' ? (
                                <><Loader2 size={12} className="animate-spin" /> Running...</>
                            ) : (
                                <><Play size={12} /> Run</>
                            )}
                        </button>
                    )}
                </div>
            </div>

            {/* ── Syntax-highlighted source code ──────────────────── */}
            <SyntaxHighlighter
                language={lang}
                style={githubGist}
                customStyle={{
                    margin: 0,
                    padding: '1rem 1.25rem',
                    fontSize: '0.8125rem',
                    lineHeight: '1.6',
                    background: '#f8f9fb',
                    borderRadius: 0,
                }}
                className="run-code-pre"
                wrapLongLines
            >
                {code}
            </SyntaxHighlighter>

            {/* ── Output area ─────────────────────────────────────── */}
            {status !== 'idle' && (
                <div className="border-t border-slate-200">
                    {/* Status bar */}
                    <div className={`flex items-center gap-2 px-4 py-2 text-xs font-medium ${status === 'running' ? 'bg-blue-50 text-blue-600' :
                            status === 'success' ? 'bg-emerald-50 text-emerald-600' :
                                'bg-red-50 text-red-600'
                        }`}>
                        {status === 'running' && <Loader2 size={12} className="animate-spin" />}
                        {status === 'success' && <CheckCircle2 size={12} />}
                        {status === 'error' && <AlertCircle size={12} />}
                        <TerminalSquare size={12} />
                        Output
                    </div>

                    {/* Stdout text */}
                    {output && (
                        <pre className="bg-slate-50 text-slate-700 p-4 text-xs overflow-x-auto whitespace-pre-wrap m-0 leading-relaxed font-mono">
                            {output}
                        </pre>
                    )}

                    {/* Matplotlib figures */}
                    {images.map((b64, i) => (
                        <div key={i} className="bg-white p-3 flex justify-center border-t border-slate-100">
                            <img
                                src={`data:image/png;base64,${b64}`}
                                alt={`Figure ${i + 1}`}
                                className="max-w-full rounded-lg shadow-sm"
                                style={{ maxHeight: '500px' }}
                            />
                        </div>
                    ))}

                    {/* Empty output notice */}
                    {status === 'success' && !output && images.length === 0 && (
                        <p className="px-4 py-3 text-xs text-slate-400 italic bg-slate-50">
                            (no output)
                        </p>
                    )}
                </div>
            )}
        </div>
    )
}
