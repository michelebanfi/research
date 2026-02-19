import { useEffect, useRef, useState } from "react";
import { Radius, Info, ChevronDown, ChevronUp, MessageSquare, Microscope } from "lucide-react";
import mermaid from "mermaid";

// ── Diagram 1: Research + Reasoning agent graph ────────────────────────────
const AGENT_GRAPH_MERMAID = `
graph TD
    start([Start]):::first --> router{Router}
    router -->|research| think
    router -->|reasoning| generate_plan

    subgraph Research
        think[Think] --> after_think{Decision}
        after_think -->|answer| finalize_research[Finalize]
        after_think -->|tool| execute_tool[Execute Tool]
        after_think -->|error| handle_no_action[Handle Error]
        execute_tool --> think
        handle_no_action --> think
    end

    subgraph Reasoning
        generate_plan[Generate Plan] --> after_plan{Has Plan?}
        after_plan -->|yes| generate_code[Generate Code]
        after_plan -->|no| finalize_reasoning[Finalize]
        generate_code --> execute_sandbox[Sandbox]
        execute_sandbox --> after_sandbox{Success?}
        after_sandbox -->|output| verify_result[Verify]
        after_sandbox -->|error| generate_code
        after_sandbox -->|max retry| finalize_reasoning
        verify_result --> after_verify{Passed?}
        after_verify -->|yes| finalize_reasoning
        after_verify -->|fail| generate_code
    end

    finalize_research --> finish([End])
    finalize_reasoning --> finish

    classDef first fill:#e0e7ff,stroke:#6366f1,stroke-width:2px,color:#1e1b4b
    classDef last fill:#fce7f3,stroke:#ec4899,stroke-width:2px,color:#831843
    classDef process fill:#f0fdf4,stroke:#22c55e,stroke-width:2px,color:#14532d
    classDef decision fill:#fef3c7,stroke:#f59e0b,stroke-width:2px,color:#92400e
    
    class start,finish first
    class think,generate_plan,generate_code,execute_tool,handle_no_action,execute_sandbox,verify_result,finalize_research,finalize_reasoning process
    class router,after_think,after_plan,after_sandbox,after_verify decision
`;

// ── Diagram 2: Paper Analyzer pipeline ────────────────────────────────────
const ANALYZE_GRAPH_MERMAID = `
graph TD
    start([Analyze Request]):::first --> load_meta[Load File Metadata]
    load_meta --> load_chunks[Fetch All Chunks]
    load_chunks --> group{Group by Section Heading}

    group --> section_loop

    subgraph PaperAnalyzer ["PaperAnalyzer — Iterative Section Loop"]
        section_loop[Next Section] --> build_prompt[Build Analysis Prompt]
        build_prompt --> llm[OpenRouter LLM]
        llm --> emit[Emit section Event]
        emit --> more{More sections?}
        more -->|yes| section_loop
        more -->|no| assemble
    end

    assemble[Assemble Full Markdown] --> complete[Emit complete Event]
    complete --> finish([Download .md]):::last

    subgraph Prompt Structure ["Per-Section Prompt"]
        p1["1 — Plain-English Summary"]
        p2["2 — Mathematical Analysis (LaTeX)"]
        p3["3 — Physical / Intuitive Meaning"]
        p4["4 — Visualization Code (matplotlib)"]
        p5["5 — Key Takeaways"]
    end

    build_prompt -.->|fills| Prompt Structure

    subgraph Frontend ["Frontend — Live Streaming"]
        ws["/ws/analyze WebSocket"]
        progress["Section Progress Cards"]
        markdown["ReactMarkdown + KaTeX"]
        runcode["RunCodeBlock (sandbox)"]
        ws --> progress
        ws --> markdown
        markdown --> runcode
    end

    emit -.->|streams| ws

    classDef first fill:#e0e7ff,stroke:#6366f1,stroke-width:2px,color:#1e1b4b
    classDef last fill:#d1fae5,stroke:#10b981,stroke-width:2px,color:#064e3b
    classDef process fill:#f0fdf4,stroke:#22c55e,stroke-width:2px,color:#14532d
    classDef decision fill:#fef3c7,stroke:#f59e0b,stroke-width:2px,color:#92400e
    classDef llm fill:#ede9fe,stroke:#8b5cf6,stroke-width:2px,color:#4c1d95

    class start first
    class finish last
    class load_meta,load_chunks,build_prompt,emit,assemble,complete,section_loop process
    class group,more decision
    class llm llm
`;

type DiagramTab = "agent" | "analyze";

const MERMAID_CONFIG = {
  startOnLoad: false,
  theme: "default" as const,
  themeVariables: {
    primaryColor: "#e0e7ff",
    primaryTextColor: "#1e1b4b",
    primaryBorderColor: "#6366f1",
    lineColor: "#64748b",
    secondaryColor: "#f0fdf4",
    tertiaryColor: "#fffbeb",
    background: "#ffffff",
    mainBkg: "#f8fafc",
    nodeBorder: "#cbd5e1",
    clusterBkg: "#f8fafc",
    clusterBorder: "#e2e8f0",
    titleColor: "#1e293b",
    edgeLabelBackground: "#ffffff",
    nodeTextColor: "#1e1b4b",
    textColor: "#1e1b4b",
    fontSize: "14px",
  },
  flowchart: {
    curve: "basis",
    padding: 20,
    nodeSpacing: 60,
    rankSpacing: 80,
    useMaxWidth: true,
  },
  securityLevel: "loose" as const,
};

export default function LangGraphTab() {
  const mermaidRef = useRef<HTMLDivElement>(null);
  const [isLegendExpanded, setIsLegendExpanded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<DiagramTab>("agent");

  const diagram = activeTab === "agent" ? AGENT_GRAPH_MERMAID : ANALYZE_GRAPH_MERMAID;
  const diagramId = activeTab === "agent" ? "langgraph-agent" : "langgraph-analyze";

  useEffect(() => {
    const renderMermaid = async () => {
      setError(null);
      try {
        mermaid.initialize(MERMAID_CONFIG);
        await new Promise((resolve) => setTimeout(resolve, 100));
        if (mermaidRef.current) {
          mermaidRef.current.innerHTML = `<div class="mermaid" id="${diagramId}">${diagram}</div>`;
          await mermaid.run({ querySelector: `#${diagramId}` });
        }
      } catch (err) {
        console.error("Failed to render mermaid diagram:", err);
        setError(`Failed to render graph: ${err}`);
      }
    };
    const timeoutId = setTimeout(renderMermaid, 50);
    return () => clearTimeout(timeoutId);
  }, [activeTab, diagram, diagramId]);

  return (
    <div className="flex h-full">
      <div className="flex-1 flex flex-col min-h-0">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-2 border-b border-border shrink-0">
          <div className="flex items-center gap-2">
            <Radius size={20} />
            <div>
              <h2 className="text-lg font-semibold">Architecture</h2>
              <p className="text-xs text-muted">Agent pipeline execution flows</p>
            </div>
          </div>

          {/* Diagram switcher tabs */}
          <div className="flex items-center gap-1 bg-slate-100 dark:bg-muted/20 p-1 rounded-lg">
            <button
              onClick={() => setActiveTab("agent")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition-all ${activeTab === "agent"
                  ? "bg-white dark:bg-surface shadow text-secondary"
                  : "text-muted hover:text-text"
                }`}
            >
              <MessageSquare size={13} />
              Chat / Reasoning
            </button>
            <button
              onClick={() => setActiveTab("analyze")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition-all ${activeTab === "analyze"
                  ? "bg-white dark:bg-surface shadow text-secondary"
                  : "text-muted hover:text-text"
                }`}
            >
              <Microscope size={13} />
              Paper Analyzer
            </button>
          </div>
        </div>

        {/* Diagram area */}
        <div className="flex-1 overflow-auto p-6 bg-slate-50 min-h-0">
          {error ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center text-red-500">
                <Info size={48} className="mx-auto mb-4" />
                <p>{error}</p>
              </div>
            </div>
          ) : (
            <div className="flex justify-center">
              <div
                ref={mermaidRef}
                className="bg-white rounded-lg shadow-lg p-8"
                style={{ minWidth: "900px", maxWidth: "1400px" }}
              />
            </div>
          )}
        </div>

        {/* Legend / Info Panel */}
        <div className="px-4 py-3 bg-surface border-t border-border shrink-0">
          <button
            onClick={() => setIsLegendExpanded(!isLegendExpanded)}
            className="flex items-center gap-2 text-sm font-medium text-secondary hover:text-secondary-hover transition-colors"
          >
            {isLegendExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            <Info size={16} />
            Node Legend &amp; Description
          </button>

          {isLegendExpanded && activeTab === "agent" && (
            <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div className="space-y-3">
                <h4 className="font-semibold text-purple-600">Research Sub-Graph</h4>
                <div className="space-y-2 text-muted">
                  <div className="flex gap-2"><span className="font-medium text-slate-700 min-w-[100px]">Think:</span><span>LLM decides next action or final answer (JSON format)</span></div>
                  <div className="flex gap-2"><span className="font-medium text-slate-700 min-w-[100px]">Execute Tool:</span><span>Runs vector_search, graph_search, web_search, etc.</span></div>
                  <div className="flex gap-2"><span className="font-medium text-slate-700 min-w-[100px]">Handle Error:</span><span>Handles parse errors from LLM responses</span></div>
                  <div className="flex gap-2"><span className="font-medium text-slate-700 min-w-[100px]">Finalize:</span><span>Builds the final response returned to the UI</span></div>
                </div>
              </div>
              <div className="space-y-3">
                <h4 className="font-semibold text-blue-600">Reasoning Sub-Graph</h4>
                <div className="space-y-2 text-muted">
                  <div className="flex gap-2"><span className="font-medium text-slate-700 min-w-[100px]">Generate Plan:</span><span>Creates goal + verification criteria before coding</span></div>
                  <div className="flex gap-2"><span className="font-medium text-slate-700 min-w-[100px]">Generate Code:</span><span>Writes Python code to solve the task</span></div>
                  <div className="flex gap-2"><span className="font-medium text-slate-700 min-w-[100px]">Sandbox:</span><span>Executes code in isolated subprocess (5s timeout)</span></div>
                  <div className="flex gap-2"><span className="font-medium text-slate-700 min-w-[100px]">Verify:</span><span>Runs assertion script to validate output</span></div>
                </div>
              </div>
              <div className="space-y-3 md:col-span-2">
                <h4 className="font-semibold text-emerald-600">Control Flow</h4>
                <div className="space-y-2 text-muted">
                  <div className="flex gap-2"><span className="font-medium text-slate-700 min-w-[100px]">Router:</span><span>Dispatches to Research or Reasoning based on mode toggle</span></div>
                  <div className="flex gap-2"><span className="font-medium text-slate-700 min-w-[100px]">Decision Nodes:</span><span>Conditional routing based on LLM output and execution results</span></div>
                </div>
              </div>
            </div>
          )}

          {isLegendExpanded && activeTab === "analyze" && (
            <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div className="space-y-3">
                <h4 className="font-semibold text-emerald-600">PaperAnalyzer Engine</h4>
                <div className="space-y-2 text-muted">
                  <div className="flex gap-2"><span className="font-medium text-slate-700 min-w-[120px]">Load &amp; Group:</span><span>Fetches all chunks from DB, groups by deepest heading in metadata</span></div>
                  <div className="flex gap-2"><span className="font-medium text-slate-700 min-w-[120px]">Section Loop:</span><span>Iterates sections in document order (sequential, not parallel)</span></div>
                  <div className="flex gap-2"><span className="font-medium text-slate-700 min-w-[120px]">LLM Call:</span><span>Calls OpenRouter with structured prompt per section (8k char budget)</span></div>
                  <div className="flex gap-2"><span className="font-medium text-slate-700 min-w-[120px]">Event Stream:</span><span>Emits start / section / complete events over /ws/analyze</span></div>
                </div>
              </div>
              <div className="space-y-3">
                <h4 className="font-semibold text-purple-600">Frontend Rendering</h4>
                <div className="space-y-2 text-muted">
                  <div className="flex gap-2"><span className="font-medium text-slate-700 min-w-[120px]">Progress Panel:</span><span>Live section cards (pending → analyzing → done)</span></div>
                  <div className="flex gap-2"><span className="font-medium text-slate-700 min-w-[120px]">ReactMarkdown:</span><span>Renders markdown with KaTeX for LaTeX equations</span></div>
                  <div className="flex gap-2"><span className="font-medium text-slate-700 min-w-[120px]">RunCodeBlock:</span><span>Python snippets have a Run button → sandbox execution → stdout + matplotlib figure</span></div>
                  <div className="flex gap-2"><span className="font-medium text-slate-700 min-w-[120px]">Download:</span><span>Exports full assembled markdown as .md file</span></div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
