import { useEffect, useRef, useState } from "react";
import { Radius, Info, ChevronDown, ChevronUp } from "lucide-react";
import mermaid from "mermaid";

// Static LangGraph architecture diagram - mirrors build_research_graph() in agent_graph.py
const GRAPH_MERMAID = `
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
    classDef clusterStyle fill:#f8fafc,stroke:#64748b,stroke-width:1px
    
    class start,finish first
    class think,generate_plan,generate_code,execute_tool,handle_no_action,execute_sandbox,verify_result,finalize_research,finalize_reasoning process
    class router,after_think,after_plan,after_sandbox,after_verify decision
`;

export default function LangGraphTab() {
  const mermaidRef = useRef<HTMLDivElement>(null);
  const [isLegendExpanded, setIsLegendExpanded] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const renderMermaid = async () => {
      try {
        // Initialize mermaid with light theme colors
        mermaid.initialize({
          startOnLoad: false,
          theme: "default",
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
          securityLevel: "loose",
        });

        // Wait for DOM to be ready
        await new Promise((resolve) => setTimeout(resolve, 100));

        if (mermaidRef.current) {
          const id = "langgraph-diagram";
          mermaidRef.current.innerHTML = `<div class="mermaid" id="${id}">${GRAPH_MERMAID}</div>`;
          await mermaid.run({
            querySelector: `#${id}`,
          });
        }
      } catch (err) {
        console.error("Failed to render mermaid diagram:", err);
        setError(`Failed to render graph diagram: ${err}`);
      }
    };

    // Small delay to ensure component is fully mounted
    const timeoutId = setTimeout(renderMermaid, 50);
    return () => clearTimeout(timeoutId);
  }, []);

  return (
    <div className="flex h-full">
      {/* Graph Area */}
      <div className="flex-1 flex flex-col">
        <div className="flex items-center justify-between px-4 py-2 border-b border-border">
          <div className="flex items-center gap-2">
            <Radius size={20} />
            <div>
              <h2 className="text-lg font-semibold">LangGraph Architecture</h2>
              <p className="text-xs text-muted">
                Agent StateGraph execution flow
              </p>
            </div>
          </div>
        </div>

        <div className="flex-1 overflow-auto p-6 bg-slate-50">
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
                style={{ minWidth: "900px", maxWidth: "1200px" }}
              />
            </div>
          )}
        </div>

        {/* Legend / Info Panel */}
        <div className="px-4 py-3 bg-surface border-t border-border">
          <button
            onClick={() => setIsLegendExpanded(!isLegendExpanded)}
            className="flex items-center gap-2 text-sm font-medium text-secondary hover:text-secondary-hover transition-colors"
          >
            {isLegendExpanded ? (
              <ChevronUp size={16} />
            ) : (
              <ChevronDown size={16} />
            )}
            <Info size={16} />
            Node Legend & Description
          </button>

          {isLegendExpanded && (
            <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div className="space-y-3">
                <h4 className="font-semibold text-purple-600">
                  Research Sub-Graph
                </h4>
                <div className="space-y-2 text-muted">
                  <div className="flex gap-2">
                    <span className="font-medium text-slate-700 min-w-[100px]">
                      Think:
                    </span>
                    <span>
                      LLM decides next action or final answer (JSON format)
                    </span>
                  </div>
                  <div className="flex gap-2">
                    <span className="font-medium text-slate-700 min-w-[100px]">
                      Execute Tool:
                    </span>
                    <span>
                      Runs vector_search, graph_search, web_search, etc.
                    </span>
                  </div>
                  <div className="flex gap-2">
                    <span className="font-medium text-slate-700 min-w-[100px]">
                      Handle Error:
                    </span>
                    <span>Handles parse errors from LLM responses</span>
                  </div>
                  <div className="flex gap-2">
                    <span className="font-medium text-slate-700 min-w-[100px]">
                      Finalize:
                    </span>
                    <span>Builds the final response returned to the UI</span>
                  </div>
                </div>
              </div>

              <div className="space-y-3">
                <h4 className="font-semibold text-blue-600">
                  Reasoning Sub-Graph
                </h4>
                <div className="space-y-2 text-muted">
                  <div className="flex gap-2">
                    <span className="font-medium text-slate-700 min-w-[100px]">
                      Generate Plan:
                    </span>
                    <span>
                      Creates goal + verification criteria before coding
                    </span>
                  </div>
                  <div className="flex gap-2">
                    <span className="font-medium text-slate-700 min-w-[100px]">
                      Generate Code:
                    </span>
                    <span>Writes Python code to solve the task</span>
                  </div>
                  <div className="flex gap-2">
                    <span className="font-medium text-slate-700 min-w-[100px]">
                      Sandbox:
                    </span>
                    <span>
                      Executes code in isolated subprocess (5s timeout)
                    </span>
                  </div>
                  <div className="flex gap-2">
                    <span className="font-medium text-slate-700 min-w-[100px]">
                      Verify:
                    </span>
                    <span>Runs assertion script to validate output</span>
                  </div>
                </div>
              </div>

              <div className="space-y-3 md:col-span-2">
                <h4 className="font-semibold text-emerald-600">Control Flow</h4>
                <div className="space-y-2 text-muted">
                  <div className="flex gap-2">
                    <span className="font-medium text-slate-700 min-w-[100px]">
                      Router:
                    </span>
                    <span>
                      Dispatches to Research or Reasoning based on mode toggle
                    </span>
                  </div>
                  <div className="flex gap-2">
                    <span className="font-medium text-slate-700 min-w-[100px]">
                      Decision Nodes:
                    </span>
                    <span>
                      Conditional routing based on LLM output and execution
                      results
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
