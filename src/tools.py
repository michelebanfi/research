"""
REQ-TOOL-01, REQ-TOOL-02, REQ-TOOL-03: Tool definitions for the Research Agent.

Each tool is a callable that takes specific inputs and returns formatted strings
for the LLM to use as observations.
"""

from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass


@dataclass
class Tool:
    """Represents a tool available to the agent."""
    name: str
    description: str
    parameters: str  # Human-readable parameter description
    execute: Callable[..., str]


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
    
    def _vector_search(self, query: str) -> str:
        """
        REQ-TOOL-01: Vector Search Tool
        
        Generates embedding for the query and searches for similar chunks.
        Returns formatted results with source information.
        """
        self._notify("vector_search", "start")
        
        try:
            # Generate embedding for the query
            query_embedding = self.ai.generate_embedding(query)
            
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
    
    def _graph_search(self, concepts: str) -> str:
        """
        REQ-TOOL-02: Graph Search Tool
        
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
                limit=10
            )
            
            if not matching_nodes:
                self._notify("graph_search", "complete")
                return f"No concepts matching '{concepts}' found in the knowledge graph."
            
            node_ids = [n['id'] for n in matching_nodes]
            matched_names = [n['name'] for n in matching_nodes]
            
            # Get 1-hop related concepts
            related_concepts = self.db.get_related_concepts(node_ids, max_hops=1)
            related_names = [n['name'] for n in related_concepts[:10]] if related_concepts else []
            
            # Expand node_ids with related concepts
            if related_concepts:
                node_ids.extend([n['id'] for n in related_concepts[:5]])
            
            # Get chunks connected to these concepts
            graph_chunks = self.db.get_chunks_by_concepts(
                node_ids,
                self.project_id,
                limit=5
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
    
    def _project_summary(self, _: str = "") -> str:
        """
        REQ-TOOL-03: Project Summary Tool
        
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
