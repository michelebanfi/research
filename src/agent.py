"""
REQ-AGENT-01, REQ-AGENT-02: Research Agent with tool-calling loop.

Implements a ReAct-style (Thought → Action → Observation) loop for tool calling,
compatible with Qwen and other models that don't have native function calling.
"""

import asyncio
import re
import time
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field

from src.reasoning_agent import ReasoningAgent
from src.models import ReasoningResponse
from src import chat_logger


@dataclass
class AgentResponse:
    """Response from the agent after processing a query."""
    answer: str
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    matched_concepts: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    model_name: str = "unknown"
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
        status_callback: Optional[Callable[[str, str], None]] = None,
        do_rerank: bool = True
    ):
        """
        Initialize the Research Agent.
        
        Args:
            ai_engine: AIEngine instance for LLM calls
            database: DatabaseClient instance
            project_id: Current project ID
            status_callback: Optional callback(tool_name, phase) for UI updates
            do_rerank: Whether to re-rank vector search results using FlashRank
        """
        self.ai = ai_engine
        self.db = database
        self.project_id = project_id
        self.status_callback = status_callback
        
        # Import here to avoid circular imports
        from src.tools import ToolRegistry
        self.tools = ToolRegistry(database, ai_engine, project_id, status_callback, do_rerank=do_rerank)
    
    def _get_system_prompt(self) -> str:
        """
        REQ-AGENT-02: Create system prompt with tool usage guidance.
        Simplified for smaller models with very explicit instructions.
        """
        tool_descriptions = self.tools.get_tool_descriptions()
        
        return f"""You are a research assistant with access to a knowledge base.

TOOLS:
{tool_descriptions}

FORMAT:
You must respond in JSON format with a "thought" field and either an "action" or "final_answer" field.

RESPONSE EXAMPLES:

1. To call a tool:
{{
  "thought": "I need to search for machine learning definitions.",
  "action": {{
    "tool_name": "vector_search",
    "argument": "machine learning definition"
  }}
}}

2. To give a final answer:
{{
  "thought": "I have found enough information to answer the user.",
  "final_answer": "Machine learning is a field of study..."
}}

GUIDELINES:
1. Always include a "thought" field to explain your reasoning step-by-step.
2. Only use valid JSON.
3. Choose the most relevant tool for the task.
4. When you have sufficient information, provide a "final_answer".
"""

    def _parse_json_response(self, response: str) -> dict:
        """
        Parses the LLM response as JSON.
        Handles code blocks and raw JSON.
        """
        import json
        try:
            # Strip markdown code blocks if present
            clean_response = response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]
            if clean_response.startswith("```"):
                clean_response = clean_response[3:]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]
            
            return json.loads(clean_response.strip())
        except json.JSONDecodeError:
            return {}

    def _parse_action(self, response_data: dict) -> Optional[tuple]:
        """
        Extracts action from parsed JSON response.
        """
        action = response_data.get("action")
        if action and isinstance(action, dict):
            return (action.get("tool_name"), action.get("argument"))
        return None
    
    def _parse_final_answer(self, response_data: dict) -> Optional[str]:
        """
        Extracts final answer from parsed JSON response.
        """
        return response_data.get("final_answer")
    
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
            # First, retrieve relevant context from knowledge base for the reasoning agent
            context = ""
            try:
                query_embedding = self.ai.generate_embedding(user_query)
                if query_embedding:
                    chunks = self.db.search_vectors(
                        query_embedding,
                        match_threshold=0.3,
                        project_id=self.project_id,
                        match_count=5
                    )
                    if chunks:
                        context_parts = [f"Document {i+1}: {c['content']}" for i, c in enumerate(chunks)]
                        context = "\n\n".join(context_parts)
            except Exception as e:
                print(f"Could not retrieve context for reasoning: {e}")
            
            reasoning_agent = ReasoningAgent(
                ai_engine=self.ai,
                status_callback=self.status_callback
            )
            return await reasoning_agent.run(task=user_query, context=context)
        
        # Start logging session for normal chat
        logger = chat_logger.new_session()
        logger.log_step("init", "Starting ResearchAgent (normal mode)", {
            "query": user_query[:200],
            "has_history": bool(chat_history),
            "max_iterations": self.MAX_ITERATIONS
        })
        
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
        conversation = f"{system_prompt}{history_text}\n\nUSER QUERY: {user_query}\n\nThink step by step in JSON format."
        
        logger.log_step("prompt", "Initial prompt built", {"prompt_length": len(conversation)})
        
        observations = []
        
        for iteration in range(self.MAX_ITERATIONS):
            try:
                logger.log_step("iteration", f"Starting iteration {iteration + 1}/{self.MAX_ITERATIONS}")
                
                # Call LLM with JSON mode
                start_time = time.time()
                llm_response_str, model_name = await self.ai._openrouter_generate(
                    conversation, 
                    return_model_name=True,
                    response_format={"type": "json_object"}
                )
                duration = time.time() - start_time
                
                logger.log_llm_call(conversation, llm_response_str, duration, self.ai.model)
                
                # Parse JSON
                response_data = self._parse_json_response(llm_response_str)
                thought = response_data.get("thought", "")
                if thought:
                     logger.log_step("thought", f"Agent thought: {thought}")

                # Check for final answer
                final_answer = self._parse_final_answer(response_data)
                if final_answer:
                    logger.log_step("complete", "Final answer found", {"answer_length": len(final_answer)})
                    logger.finish("success", final_answer[:200])
                    return AgentResponse(
                        answer=final_answer,
                        retrieved_chunks=retrieved_chunks,
                        matched_concepts=matched_concepts,
                        tools_used=tools_used,
                        model_name=model_name
                    )
                
                # Check for action
                action = self._parse_action(response_data)
                if action:
                    tool_name, argument = action
                    # Ensure argument is a string for logging/execution compatibility
                    if not isinstance(argument, str):
                        import json
                        argument = json.dumps(argument) if argument is not None else ""
                        
                    logger.log_step("action", f"Parsed action: {tool_name}", {"argument": argument[:100]})
                    tool = self.tools.get_tool(tool_name)
                    
                    if tool:
                        tools_used.append(tool_name)
                        
                        # Execute the tool (REQ-FIX-01: await async tool)
                        tool_start = time.time()
                        observation = await tool.execute(argument)
                        tool_duration = time.time() - tool_start
                        observations.append((tool_name, observation))
                        
                        logger.log_step("tool_result", f"Tool {tool_name} executed", {
                            "duration_s": round(tool_duration, 2),
                            "result_length": len(observation)
                        })
                        
                        # Extract chunks for context panel (from vector_search)
                        if tool_name == "vector_search":
                            # Re-run search to get structured results for UI
                            query_embedding = self.ai.generate_embedding(argument)
                            if query_embedding:
                                chunks = self.db.search_vectors(
                                    query_embedding,
                                    match_threshold=0.3,
                                    project_id=self.project_id,
                                    match_count=10 if self.tools.do_rerank else 5
                                )
                                # Apply same re-ranking as the tool
                                if self.tools.do_rerank and len(chunks) > 1:
                                    try:
                                        chunks = self.ai.rerank_results(argument, chunks, top_k=5)
                                    except Exception as e:
                                        print(f"Re-ranking for UI failed: {e}")
                                for c in chunks:
                                    c['source'] = 'vector'
                                retrieved_chunks.extend(chunks)
                            
                            # REQ-AGENT-03: Autonomous web search fallback
                            # Check if best relevance score is below threshold
                            best_score = 0.0
                            if chunks:
                                best_score = max(
                                    c.get('rerank_score', c.get('similarity', 0)) 
                                    for c in chunks
                                )
                            
                            if best_score < 0.3 and "web_search" not in tools_used:
                                logger.log_step("auto_fallback", f"Low relevance ({best_score:.2f}), triggering web_search")
                                web_tool = self.tools.get_tool("web_search")
                                if web_tool:
                                    web_result = await web_tool.execute(argument)
                                    observations.append(("web_search", web_result))
                                    tools_used.append("web_search")
                                    observation += f"\n\n[AUTO-FALLBACK: Low local relevance, web search results:]\n{web_result}"
                        
                        elif tool_name == "graph_search":
                            # Parse concepts for UI display
                            concepts = [c.strip() for c in argument.split(",")]
                            matched_concepts.extend(concepts)
                        
                        # Add observation to conversation
                        # Keep conversation string simple for next turn
                        conversation += f"\n\nASSISTANT JSON:\n{llm_response_str}\n\nOBSERVATION from {tool_name}:\n{observation}\n\nBased on this observation, provide the next step in JSON format."
                    else:
                        # Unknown tool
                        conversation += f"\n\nASSISTANT JSON:\n{llm_response_str}\n\nOBSERVATION: Unknown tool '{tool_name}'. Available tools are: vector_search, graph_search, project_summary.\n\nPlease try again with a valid tool in JSON format."
                else:
                    # No action found - nudge/error
                    logger.log_step("no_action", "No action parsed from JSON", {"response_preview": llm_response_str[:100]})
                    conversation += f"\n\nASSISTANT JSON:\n{llm_response_str}\n\nERROR: Invalid JSON or missing 'action'/'final_answer'. Please respond with valid JSON containing 'thought' and 'action' OR 'final_answer'."
                    
            except Exception as e:
                logger.log_error(f"Exception in iteration: {e}")
                logger.finish("error", str(e))
                return AgentResponse(
                    answer=f"I encountered an error while processing your question: {str(e)}",
                    error=str(e),
                    tools_used=tools_used
                )
        
        # Max iterations reached - use direct RAG as fallback
        logger.log_step("fallback", "Max iterations reached, trying fallback")
        if observations:
            # We have some tool results, try to synthesize an answer
            context_text = "\n\n".join([f"{name}: {obs}" for name, obs in observations])
            try:
                fallback_prompt = f"""Based on this information from my knowledge base:

{context_text}

Answer this question: {user_query}

Provide a helpful, complete answer based on the information above."""
                start_time = time.time()
                fallback_response, model_name = await self.ai._openrouter_generate(fallback_prompt, return_model_name=True)
                duration = time.time() - start_time
                logger.log_llm_call(fallback_prompt, fallback_response, duration, self.ai.model)
                logger.finish("success", "Fallback synthesis")
                return AgentResponse(
                    answer=fallback_response,
                    retrieved_chunks=retrieved_chunks,
                    matched_concepts=matched_concepts,
                    tools_used=tools_used
                )
            except Exception as e:
                logger.log_error(f"Fallback synthesis failed: {e}")
                pass
        
        # Final fallback - do a simple vector search and answer
        try:
            query_embedding = self.ai.generate_embedding(user_query)
            if query_embedding:
                chunks = self.db.search_vectors(
                    query_embedding,
                    match_threshold=0.3,
                    project_id=self.project_id,
                    match_count=5
                )
                if chunks:
                    response = await self.ai.chat_with_context_async(
                        query=user_query,
                        context_chunks=chunks,
                        chat_history=chat_history
                    )
                    return AgentResponse(
                        answer=response["response"],
                        retrieved_chunks=chunks,
                        matched_concepts=matched_concepts,
                        tools_used=["vector_search (fallback)"],
                        model_name=response.get("model", "unknown")
                    )
        except Exception as e:
            pass
        
        return AgentResponse(
            answer="I wasn't able to find relevant information to answer your question. Please try rephrasing or asking about something in your knowledge base.",
            retrieved_chunks=retrieved_chunks,
            matched_concepts=matched_concepts,
            tools_used=tools_used,
            error="Max iterations reached"
        )
