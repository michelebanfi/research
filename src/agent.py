"""
REQ-AGENT-01, REQ-AGENT-02: Research Agent with tool-calling loop.

Implements a ReAct-style (Thought → Action → Observation) loop for tool calling,
compatible with Qwen and other models that don't have native function calling.
"""

import asyncio
import re
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field

from src.reasoning_agent import ReasoningAgent
from src.models import ReasoningResponse


@dataclass
class AgentResponse:
    """Response from the agent after processing a query."""
    answer: str
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    matched_concepts: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    error: Optional[str] = None


class ResearchAgent:
    """
    Research Agent with ReAct-style tool-calling loop.
    
    Manages the think-act-observe cycle:
    1. Send user query + tool definitions to LLM
    2. Check if LLM wants to call a tool
    3. Execute tool and get observation
    4. Feed result back to LLM
    5. Repeat until FINAL ANSWER
    """
    
    MAX_ITERATIONS = 5  # Prevent infinite loops
    
    def __init__(
        self, 
        ai_engine, 
        database, 
        project_id: str,
        status_callback: Optional[Callable[[str, str], None]] = None
    ):
        """
        Initialize the Research Agent.
        
        Args:
            ai_engine: AIEngine instance for LLM calls
            database: DatabaseClient instance
            project_id: Current project ID
            status_callback: Optional callback(tool_name, phase) for UI updates
        """
        self.ai = ai_engine
        self.db = database
        self.project_id = project_id
        self.status_callback = status_callback
        
        # Import here to avoid circular imports
        from src.tools import ToolRegistry
        self.tools = ToolRegistry(database, ai_engine, project_id, status_callback)
    
    def _get_system_prompt(self) -> str:
        """
        REQ-AGENT-02: Create system prompt with tool usage guidance.
        Enhanced with explicit examples for smaller models.
        """
        tool_descriptions = self.tools.get_tool_descriptions()
        
        return f"""You are a research assistant. You MUST use tools to answer questions. Do NOT answer without using a tool first.

AVAILABLE TOOLS:
{tool_descriptions}

RULES:
1. You MUST call a tool before answering
2. For "What is this project about?" -> use project_summary
3. For specific code/concept questions -> use vector_search
4. For concept relationships -> use graph_search

FORMAT:
To call a tool, write:
ACTION: tool_name(argument)

After seeing the result, write:
FINAL ANSWER: your answer based on the tool result

EXAMPLE 1:
User: What is this project about?
ACTION: project_summary()

EXAMPLE 2:
User: How does vector search work?
ACTION: vector_search(vector search implementation)

EXAMPLE 3:
User: What concepts relate to embeddings?
ACTION: graph_search(embeddings)

NOW RESPOND TO THE USER. Start with ACTION:"""

    def _parse_action(self, response: str) -> Optional[tuple]:
        """
        REQ-FIX-02: Parse the LLM response for an ACTION: tool_name(argument) pattern.
        Uses balanced parentheses matching to handle arguments containing parens.
        
        Returns:
            Tuple of (tool_name, argument) or None if no action found
        """
        import re
        
        # First, find ACTION: tool_name( pattern
        action_start = re.search(r'ACTION:\s*(\w+)\(', response, re.IGNORECASE)
        if not action_start:
            return None
        
        tool_name = action_start.group(1).lower().strip()
        start_idx = action_start.end()  # Position right after the opening (
        
        # REQ-FIX-02: Use balanced parentheses counter to find matching close paren
        paren_count = 1
        end_idx = start_idx
        
        while end_idx < len(response) and paren_count > 0:
            char = response[end_idx]
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            end_idx += 1
        
        if paren_count != 0:
            # Unbalanced parens - fallback to simple regex
            pattern = r'ACTION:\s*(\w+)\((.*)\)'
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                return (match.group(1).lower().strip(), match.group(2).strip().strip('"').strip("'"))
            return None
        
        # Extract argument (excluding the final closing paren)
        argument = response[start_idx:end_idx-1].strip().strip('"').strip("'")
        return (tool_name, argument)
    
    def _parse_final_answer(self, response: str) -> Optional[str]:
        """
        Parse the LLM response for a FINAL ANSWER.
        
        Returns:
            The final answer text or None if not found
        """
        pattern = r'FINAL ANSWER:\s*(.+)'
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        return None
    
    async def run(
        self, 
        user_query: str, 
        chat_history: Optional[List[Dict[str, str]]] = None,
        reasoning_mode: bool = False
    ) -> Union[AgentResponse, ReasoningResponse]:
        """
        Run the agent loop to answer the user's query.
        
        Args:
            user_query: The user's question
            chat_history: Optional conversation history
            reasoning_mode: If True, use Plan & Code loop (REQ-POETIQ-02)
            
        Returns:
            AgentResponse or ReasoningResponse with the answer and metadata
        """
        # REQ-POETIQ-02: Reasoning Toggle
        if reasoning_mode:
            reasoning_agent = ReasoningAgent(
                ai_engine=self.ai,
                status_callback=self.status_callback
            )
            return await reasoning_agent.run(task=user_query)
        tools_used = []
        retrieved_chunks = []
        matched_concepts = []
        
        # Build initial prompt with history
        history_text = ""
        if chat_history:
            history_parts = []
            for msg in chat_history[-5:]:  # Last 5 messages for context
                role = "User" if msg['role'] == 'user' else "Assistant"
                history_parts.append(f"{role}: {msg['content']}")
            if history_parts:
                history_text = "\n\nRECENT CONVERSATION:\n" + "\n".join(history_parts)
        
        # Build the conversation for the agent
        system_prompt = self._get_system_prompt()
        conversation = f"{system_prompt}{history_text}\n\nUSER QUERY: {user_query}\n\nThink step by step. What tool(s) do you need to answer this question?"
        
        observations = []
        
        for iteration in range(self.MAX_ITERATIONS):
            try:
                # Call LLM
                response = await self.ai.async_client.generate(
                    model=self.ai.model,
                    prompt=conversation
                )
                llm_response = response["response"]
                
                # Check for final answer
                final_answer = self._parse_final_answer(llm_response)
                if final_answer:
                    return AgentResponse(
                        answer=final_answer,
                        retrieved_chunks=retrieved_chunks,
                        matched_concepts=matched_concepts,
                        tools_used=tools_used
                    )
                
                # Check for action
                action = self._parse_action(llm_response)
                if action:
                    tool_name, argument = action
                    tool = self.tools.get_tool(tool_name)
                    
                    if tool:
                        tools_used.append(tool_name)
                        
                        # Execute the tool (REQ-FIX-01: await async tool)
                        observation = await tool.execute(argument)
                        observations.append((tool_name, observation))
                        
                        # Extract chunks for context panel (from vector_search)
                        if tool_name == "vector_search":
                            # Re-run search to get structured results for UI
                            query_embedding = self.ai.generate_embedding(argument)
                            if query_embedding:
                                chunks = self.db.search_vectors(
                                    query_embedding,
                                    match_threshold=0.3,
                                    project_id=self.project_id,
                                    match_count=5
                                )
                                for c in chunks:
                                    c['source'] = 'vector'
                                retrieved_chunks.extend(chunks)
                        
                        elif tool_name == "graph_search":
                            # Parse concepts for UI display
                            concepts = [c.strip() for c in argument.split(",")]
                            matched_concepts.extend(concepts)
                        
                        # Add observation to conversation
                        conversation += f"\n\n{llm_response}\n\nOBSERVATION from {tool_name}:\n{observation}\n\nBased on this observation, what's next? Use another tool or provide FINAL ANSWER:"
                    else:
                        # Unknown tool
                        conversation += f"\n\n{llm_response}\n\nOBSERVATION: Unknown tool '{tool_name}'. Available tools are: vector_search, graph_search, project_summary.\n\nPlease try again or provide FINAL ANSWER:"
                else:
                    # No action found - nudge towards final answer
                    if "final" in llm_response.lower() or iteration >= self.MAX_ITERATIONS - 2:
                        # LLM might have just answered without the prefix
                        return AgentResponse(
                            answer=llm_response,
                            retrieved_chunks=retrieved_chunks,
                            matched_concepts=matched_concepts,
                            tools_used=tools_used
                        )
                    
                    conversation += f"\n\n{llm_response}\n\nRemember: Use ACTION: tool_name(argument) to call a tool, or FINAL ANSWER: to provide your response."
                    
            except Exception as e:
                return AgentResponse(
                    answer=f"I encountered an error while processing your question: {str(e)}",
                    error=str(e),
                    tools_used=tools_used
                )
        
        # Max iterations reached - return what we have
        return AgentResponse(
            answer="I wasn't able to complete the analysis within the allowed iterations. Please try rephrasing your question.",
            retrieved_chunks=retrieved_chunks,
            matched_concepts=matched_concepts,
            tools_used=tools_used,
            error="Max iterations reached"
        )
