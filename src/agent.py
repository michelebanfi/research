"""
REQ-AGENT-01, REQ-AGENT-02: Research Agent — thin wrapper around LangGraph.

The agent loop logic now lives in agent_graph.py (StateGraph).
This module keeps the public API stable:
  - ResearchAgent.__init__(ai_engine, database, project_id, ...)
  - ResearchAgent.run(user_query, chat_history, reasoning_mode) -> AgentResponse | ReasoningResponse
"""

import asyncio
import uuid
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field

from src.models import ReasoningResponse


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
    Research Agent — delegates to LangGraph StateGraph.

    Manages initialization of the graph with required dependencies and
    preserves the existing public API consumed by the UI (chat.py).
    """

    def __init__(
        self,
        ai_engine,
        database,
        project_id: str,
        status_callback: Optional[Callable[[str, str], None]] = None,
        do_rerank: bool = True,
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.ai = ai_engine
        self.db = database
        self.project_id = project_id
        self.status_callback = status_callback
        self.event_callback = event_callback

        # Import here to avoid circular imports
        from src.tools import ToolRegistry
        self.tools = ToolRegistry(
            database, ai_engine, project_id, status_callback, do_rerank=do_rerank,
        )

        # Build the LangGraph
        from src.agent_graph import build_research_graph
        self._graph = build_research_graph(
            ai_engine=ai_engine,
            db=database,
            project_id=project_id,
            tools_registry=self.tools,
            event_callback=event_callback,
        )

    async def run(
        self,
        user_query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        reasoning_mode: bool = False,
    ) -> Union[AgentResponse, ReasoningResponse]:
        """
        Run the agent graph to answer the user's query.

        Returns AgentResponse (research mode) or ReasoningResponse (reasoning mode).
        """
        initial_state = {
            "user_query": user_query,
            "chat_history": chat_history or [],
            "reasoning_mode": reasoning_mode,
        }

        # Each invocation gets a unique thread_id for checkpointing
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        final_state = await self._graph.ainvoke(initial_state, config=config)

        # The graph sets `result` in the finalize nodes
        result = final_state.get("result")

        if result is not None:
            return result

        # Fallback (should not happen if graph is correct)
        return AgentResponse(
            answer="Agent completed without producing a result.",
            error="No result in final state",
        )
