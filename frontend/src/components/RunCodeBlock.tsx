/**
 * RunCodeBlock — renders a Python code block with an inline "Run" button.
 * On click it calls POST /api/run-code, shows stdout, and embeds any
 * matplotlib figures as <img> tags underneath.
 */

import { useState } from 'react'
import { api } from '../services/api'
import { Play, Loader2, TerminalSquare, CheckCircle2, AlertCircle, RefreshCw } from 'lucide-react'

interface Props {
    code: string            // raw Python source
    language: string        // e.g. "python"
}

export default function RunCodeBlock({ code, language }: Props) {
    const [status, setStatus] = useState<'idle' | 'running' | 'success' | 'error'>('idle')
    const [output, setOutput] = useState<string>('')
    const [images, setImages] = useState<string[]>([])

    const isPython = language === 'python' || language === 'py'

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

    return (
        <div className="my-3 rounded-lg border border-border overflow-hidden">
            {/* Code block header */}
            <div className="flex items-center justify-between px-3 py-1.5 bg-slate-800 text-slate-300 text-xs">
                <span className="font-mono opacity-70">{language || 'code'}</span>
                <div className="flex items-center gap-2">
                    {status !== 'idle' && (
                        <button
                            onClick={handleReset}
                            className="flex items-center gap-1 px-2 py-0.5 rounded text-xs text-slate-400 hover:text-white transition-colors"
                            title="Reset output"
                        >
                            <RefreshCw size={11} />
                            Reset
                        </button>
                    )}
                    {isPython && (
                        <button
                            onClick={handleRun}
                            disabled={status === 'running'}
                            className="flex items-center gap-1.5 px-2.5 py-1 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-60 text-white rounded text-xs font-medium transition-colors"
                        >
                            {status === 'running' ? (
                                <><Loader2 size={11} className="animate-spin" /> Running...</>
                            ) : (
                                <><Play size={11} /> Run</>
                            )}
                        </button>
                    )}
                </div>
            </div>

            {/* Source code */}
            <pre className="bg-slate-900 p-3 overflow-x-auto text-sm text-slate-100 m-0">
                <code>{code}</code>
            </pre>

            {/* Output area */}
            {status !== 'idle' && (
                <div className="border-t border-slate-700">
                    {/* Status bar */}
                    <div className={`flex items-center gap-2 px-3 py-1.5 text-xs font-medium ${status === 'running' ? 'bg-blue-950 text-blue-300' :
                            status === 'success' ? 'bg-emerald-950 text-emerald-300' :
                                'bg-red-950 text-red-300'
                        }`}>
                        {status === 'running' && <Loader2 size={12} className="animate-spin" />}
                        {status === 'success' && <CheckCircle2 size={12} />}
                        {status === 'error' && <AlertCircle size={12} />}
                        <TerminalSquare size={12} className="ml-1" />
                        Output
                    </div>

                    {/* Stdout text */}
                    {output && (
                        <pre className="bg-slate-950 text-slate-200 p-3 text-xs overflow-x-auto whitespace-pre-wrap m-0 leading-relaxed">
                            {output}
                        </pre>
                    )}

                    {/* Matplotlib figures */}
                    {images.map((b64, i) => (
                        <div key={i} className="bg-white p-2 flex justify-center">
                            <img
                                src={`data:image/png;base64,${b64}`}
                                alt={`Figure ${i + 1}`}
                                className="max-w-full rounded shadow-sm"
                                style={{ maxHeight: '500px' }}
                            />
                        </div>
                    ))}

                    {/* Empty output notice */}
                    {status === 'success' && !output && images.length === 0 && (
                        <p className="px-3 py-2 text-xs text-slate-400 italic bg-slate-950">
                            (no output)
                        </p>
                    )}
                </div>
            )}
        </div>
    )
}
