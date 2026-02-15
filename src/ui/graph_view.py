"""
LangGraph visualization component for the UI.

Renders the agent StateGraph as a Mermaid diagram
using mermaid.js loaded from CDN inside a Streamlit HTML component.
"""

import streamlit as st
import streamlit.components.v1 as components


# The static graph architecture — mirrors build_research_graph() in agent_graph.py
_GRAPH_MERMAID = """
graph TD
    start(["Start"]):::first --> router{"Router"}
    router -->|"research"| think
    router -->|"reasoning"| generate_plan

    subgraph Research
        think["Think"] --> after_think{"Decision"}
        after_think -->|"final answer"| finalize_research["Finalize"]
        after_think -->|"tool call"| execute_tool["Execute Tool"]
        after_think -->|"parse error"| handle_no_action["Handle Error"]
        execute_tool --> think
        handle_no_action --> think
    end

    subgraph Reasoning
        generate_plan["Generate Plan"] --> after_plan{"Has Plan?"}
        after_plan -->|"yes"| generate_code["Generate Code"]
        after_plan -->|"no"| finalize_reasoning
        generate_code --> execute_sandbox["Sandbox"]
        execute_sandbox --> after_sandbox{"Success?"}
        after_sandbox -->|"output"| verify_result["Verify"]
        after_sandbox -->|"error"| generate_code
        after_sandbox -->|"max retries"| finalize_reasoning
        verify_result --> after_verify{"Passed?"}
        after_verify -->|"yes"| finalize_reasoning["Finalize"]
        after_verify -->|"fail"| generate_code
    end

    finalize_research --> finish(["End"])
    finalize_reasoning --> finish

    style start fill-opacity:0,stroke:#333,stroke-width:2px;
    style finish fill:#bfb6fc,stroke:#333,stroke-width:2px;
"""


def render_graph_tab(graph=None):
    """
    Render the agent graph architecture as a Mermaid diagram.

    If a compiled graph is passed, uses its live structure.
    Otherwise renders the static architecture diagram.
    """
    st.markdown("### 🏗️ Agent Graph Architecture")
    st.caption("This is the LangGraph StateGraph that powers the research agent.")

    mermaid_str = None

    # Try live graph first
    if graph is not None:
        try:
            mermaid_str = graph.get_graph().draw_mermaid()
        except Exception:
            pass

    # Fall back to static diagram
    if mermaid_str is None:
        mermaid_str = _GRAPH_MERMAID

    _render_mermaid(mermaid_str)

    # Legend
    with st.expander("📖 Legend", expanded=False):
        st.markdown("""
| Node | Description |
|------|-------------|
| **Router** | Dispatches to Research or Reasoning based on mode toggle |
| **Think** | LLM decides next action or final answer (JSON format) |
| **Execute Tool** | Runs vector_search, graph_search, web_search, etc. |
| **Generate Plan** | Creates goal + verification criteria before coding |
| **Generate Code** | Writes Python code to solve the task |
| **Sandbox** | Executes code in isolated subprocess (5s timeout) |
| **Verify** | Runs assertion script to validate output |
| **Finalize** | Builds the response returned to the UI |
""")


def _render_mermaid(mermaid_code: str, height: int = 600):
    """Render a Mermaid diagram using mermaid.js via an HTML component."""

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
        <style>
            body {{
                margin: 0;
                padding: 16px;
                background: transparent;
                display: flex;
                justify-content: center;
                align-items: flex-start;
                min-height: 100%;
            }}
            #graph-container {{
                width: 100%;
                overflow: auto;
            }}
            .mermaid {{
                display: flex;
                justify-content: center;
            }}
            .mermaid .node rect,
            .mermaid .node polygon,
            .mermaid .node circle {{
                stroke-width: 2px;
            }}
        </style>
    </head>
    <body>
        <div id="graph-container">
            <pre class="mermaid">
{mermaid_code}
            </pre>
        </div>
        <script>
            mermaid.initialize({{
                startOnLoad: true,
                theme: 'base',
                themeVariables: {{
                    primaryColor: '#6C5CE7',
                    primaryTextColor: '#fff',
                    primaryBorderColor: '#5A4BD1',
                    lineColor: '#a29bfe',
                    secondaryColor: '#00b894',
                    tertiaryColor: '#f8f9fa',
                    background: '#1e1e2e',
                    mainBkg: '#6C5CE7',
                    nodeBorder: '#5A4BD1',
                    clusterBkg: '#2d2d44',
                    clusterBorder: '#44475a',
                    titleColor: '#f8f8f2',
                    edgeLabelBackground: '#2d2d44',
                    fontSize: '14px'
                }},
                flowchart: {{
                    curve: 'basis',
                    padding: 20,
                    nodeSpacing: 50,
                    rankSpacing: 60,
                    htmlLabels: true
                }}
            }});
        </script>
    </body>
    </html>
    """
    components.html(html, height=height, scrolling=True)
