"""
REQ-TOOL-01, REQ-TOOL-02, REQ-TOOL-03: Tool definitions for the Research Agent.

Each tool is a callable that takes specific inputs and returns formatted strings
for the LLM to use as observations.

REQ-FIX-01: All tool execute methods are now async-native.
"""

from typing import List, Dict, Any, Callable, Optional, Awaitable
from dataclasses import dataclass

from src.sandbox import run_code


@dataclass
class Tool:
    """Represents a tool available to the agent."""
    name: str
    description: str
    parameters: str  # Human-readable parameter description
    execute: Callable[..., Awaitable[str]]  # REQ-FIX-01: Async callable


class ToolRegistry:
    """
    Registry of tools available to the Research Agent.
    Requires db and ai_engine instances to be set before use.
    """
    
    def __init__(self, database, ai_engine, project_id: str, status_callback: Optional[Callable] = None):
        """
        Initialize the tool registry with dependencies.
        
        Args:
            database: DatabaseClient instance
            ai_engine: AIEngine instance  
            project_id: Current project ID for scoped queries
            status_callback: Optional callback(tool_name, phase) for UI updates
        """
        self.db = database
        self.ai = ai_engine
        self.project_id = project_id
        self.status_callback = status_callback
        self._tools = self._create_tools()
    
    def _notify(self, tool_name: str, phase: str):
        """Notify UI of tool execution status."""
        if self.status_callback:
            self.status_callback(tool_name, phase)
    
    def _create_tools(self) -> Dict[str, Tool]:
        """Create and return all available tools."""
        return {
            "vector_search": Tool(
                name="vector_search",
                description="Search the knowledge base for specific information using semantic similarity. Use this for questions about specific code, concepts, facts, or technical details.",
                parameters="query (string): The search query to find relevant information",
                execute=self._vector_search
            ),
            "graph_search": Tool(
                name="graph_search",
                description="Search the knowledge graph for concept connections and relationships. Use this to find how concepts are related or to explore the structure of the knowledge.",
                parameters="concepts (comma-separated string): Concept names to search for and find connections",
                execute=self._graph_search
            ),
            "project_summary": Tool(
                name="project_summary",
                description="Get a high-level overview of all files and their summaries in the project. Use this for broad questions like 'What is this project about?' or 'Give me an overview'.",
                parameters="None required",
                execute=self._project_summary
            ),
            "python_interpreter": Tool(
                name="python_interpreter",
                description="Execute Python code and return the output. Use this to run computations, process data, or verify results. The code runs in an isolated environment with common libraries available (json, math, re, collections, itertools).",
                parameters="code (string): Python code to execute",
                execute=self._python_interpreter
            ),
        }
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def get_all_tools(self) -> List[Tool]:
        """Get all available tools."""
        return list(self._tools.values())
    
    def get_tool_descriptions(self) -> str:
        """Get formatted tool descriptions for the system prompt."""
        lines = []
        for i, tool in enumerate(self._tools.values(), 1):
            lines.append(f"{i}. {tool.name}({tool.parameters})")
            lines.append(f"   - {tool.description}")
        return "\n".join(lines)
    
    # ==================== TOOL IMPLEMENTATIONS ====================
    
    async def _vector_search(self, query: str) -> str:
        """
        REQ-TOOL-01: Vector Search Tool
        REQ-FIX-01: Now async-native.
        
        Generates embedding for the query and searches for similar chunks.
        Returns formatted results with source information.
        """
        self._notify("vector_search", "start")
        
        try:
            # Generate embedding for the query (async)
            query_embedding = await self.ai.generate_embedding_async(query)
            
            if not query_embedding:
                self._notify("vector_search", "complete")
                return "Error: Could not generate embedding for query."
            
            # Search vectors
            results = self.db.search_vectors(
                query_embedding,
                match_threshold=0.3,
                project_id=self.project_id,
                match_count=5
            )
            
            if not results:
                self._notify("vector_search", "complete")
                return "No relevant results found in the knowledge base."
            
            # Format results
            output_parts = [f"Found {len(results)} relevant chunks:\n"]
            
            for i, chunk in enumerate(results, 1):
                similarity = chunk.get('similarity', 0)
                content = chunk.get('content', '')[:500]  # Truncate for LLM context
                file_path = chunk.get('file_path', 'Unknown source')
                
                output_parts.append(f"[Source {i}] (similarity: {similarity:.2f})")
                output_parts.append(f"File: {file_path}")
                output_parts.append(f"Content: {content}")
                output_parts.append("---")
            
            self._notify("vector_search", "complete")
            return "\n".join(output_parts)
            
        except Exception as e:
            self._notify("vector_search", "complete")
            return f"Error during vector search: {str(e)}"
    
    async def _graph_search(self, concepts: str) -> str:
        """
        REQ-TOOL-02: Graph Search Tool
        REQ-FIX-01: Now async-native.
        REQ-IMP-04: Increased limits for denser graph retrieval.
        
        Searches for nodes matching the concept names, finds related concepts,
        and retrieves connected chunks.
        """
        self._notify("graph_search", "start")
        
        try:
            # Parse concepts from comma-separated string
            concept_list = [c.strip().lower() for c in concepts.split(",") if c.strip()]
            
            if not concept_list:
                self._notify("graph_search", "complete")
                return "Error: No concepts provided. Please specify concept names."
            
            # Search for matching nodes
            matching_nodes = self.db.search_nodes_by_name(
                concept_list,
                self.project_id,
                limit=20  # REQ-IMP-04: Increased from 10
            )
            
            if not matching_nodes:
                self._notify("graph_search", "complete")
                return f"No concepts matching '{concepts}' found in the knowledge graph."
            
            node_ids = [n['id'] for n in matching_nodes]
            matched_names = [n['name'] for n in matching_nodes]
            
            # Get 1-hop related concepts
            related_concepts = self.db.get_related_concepts(node_ids, max_hops=1)
            related_names = [n['name'] for n in related_concepts[:15]] if related_concepts else []  # REQ-IMP-04: Increased
            
            # Expand node_ids with related concepts
            if related_concepts:
                node_ids.extend([n['id'] for n in related_concepts[:10]])  # REQ-IMP-04: Increased from 5
            
            # Get chunks connected to these concepts
            graph_chunks = self.db.get_chunks_by_concepts(
                node_ids,
                self.project_id,
                limit=20  # REQ-IMP-04: Increased from 5
            )
            
            # Format output
            output_parts = []
            
            output_parts.append(f"**Matched Concepts:** {', '.join(matched_names)}")
            
            if related_names:
                output_parts.append(f"**Related Concepts (1-hop):** {', '.join(related_names)}")
            
            if graph_chunks:
                output_parts.append(f"\n**Connected Knowledge ({len(graph_chunks)} chunks):**\n")
                for i, chunk in enumerate(graph_chunks, 1):
                    content = chunk.get('content', '')[:400]
                    file_path = chunk.get('file_path', 'Unknown')
                    output_parts.append(f"[Chunk {i}] from {file_path}")
                    output_parts.append(f"{content}")
                    output_parts.append("---")
            else:
                output_parts.append("\nNo document chunks directly connected to these concepts.")
            
            self._notify("graph_search", "complete")
            return "\n".join(output_parts)
            
        except Exception as e:
            self._notify("graph_search", "complete")
            return f"Error during graph search: {str(e)}"
    
    async def _project_summary(self, _: str = "") -> str:
        """
        REQ-TOOL-03: Project Summary Tool
        REQ-FIX-01: Now async-native.
        
        Retrieves all file summaries in the project for a high-level overview.
        """
        self._notify("project_summary", "start")
        
        try:
            summaries = self.db.get_all_file_summaries(self.project_id)
            
            if not summaries or summaries == "No files found in this project.":
                self._notify("project_summary", "complete")
                return "This project has no files ingested yet."
            
            self._notify("project_summary", "complete")
            return f"**Project Overview:**\n\n{summaries}"
            
        except Exception as e:
            self._notify("project_summary", "complete")
            return f"Error getting project summary: {str(e)}"
    
    async def _python_interpreter(self, code: str) -> str:
        """
        REQ-POETIQ-01: Python Interpreter Tool
        REQ-FIX-01: Now async-native - no blocking asyncio.run().
        
        Executes Python code in a secure sandbox and returns the output.
        """
        self._notify("python_interpreter", "start")
        
        if not code or not code.strip():
            self._notify("python_interpreter", "complete")
            return "Error: No code provided to execute."
        
        try:
            # REQ-FIX-01: Directly await async sandbox (no blocking asyncio.run)
            success, output = await run_code(code, timeout_s=5.0)
            
            self._notify("python_interpreter", "complete")
            
            if success:
                if output:
                    return f"**Execution successful:**\n```\n{output}\n```"
                else:
                    return "**Execution successful** (no output)"
            else:
                return f"**Execution failed:**\n```\n{output}\n```"
                
        except Exception as e:
            self._notify("python_interpreter", "complete")
            return f"Error executing code: {str(e)}"
