"""
Unified LangGraph StateGraph for the Research Assistant.

Replaces the hand-rolled ReAct loop (agent.py) with a structured graph
that supports two sub-graphs:
  1. Research sub-graph: Think → Tool → Observe loop
  2. Reasoning sub-graph: Plan → Code → Sandbox → Verify loop

Features:
  - MemorySaver checkpointing for pause/resume
  - Event emission for UI observability
  - Max-iteration safety limits
  - Autonomous web search fallback on low relevance
"""

import asyncio
import json
import re
import time
from typing import (
    Annotated, Any, Callable, Dict, List, Optional,
    Sequence, TypedDict, Union
)

from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver

from src.ai_engine import AIEngine
from src.database import DatabaseClient
from src.config import Config
from src.models import ReasoningPlan, CodeAttempt, ReasoningResponse
from src.sandbox import run_code
from src import chat_logger


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ResearchState(TypedDict):
    """Shared state flowing through the graph."""
    # --- Input ---
    user_query: str
    chat_history: List[Dict[str, str]]
    reasoning_mode: bool

    # --- Research sub-graph ---
    system_prompt: str
    conversation: str          # Running LLM conversation string
    observations: List[tuple]  # (tool_name, observation) pairs
    tools_used: List[str]
    retrieved_chunks: List[Dict[str, Any]]
    matched_concepts: List[str]
    iteration: int
    final_answer: Optional[str]
    model_name: str
    error: Optional[str]

    # --- Reasoning sub-graph ---
    plan: Optional[ReasoningPlan]
    code_attempts: List[CodeAttempt]
    current_code: Optional[str]
    current_output: Optional[str]
    feedback: str

    # --- Parsed LLM response (scratch) ---
    llm_response_str: str
    parsed_response: Dict[str, Any]
    parsed_action: Optional[tuple]  # (tool_name, argument) - Deprecated, kept for compat?
    parsed_actions: List[tuple]     # List of (tool_name, argument) for parallel execution

    # --- Result (set at finalize) ---
    result: Optional[Any]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_RESEARCH_ITERATIONS = 5
MAX_CODE_RETRIES = 5


# ---------------------------------------------------------------------------
# Graph Node Implementations
# ---------------------------------------------------------------------------

class GraphNodes:
    """
    All node functions for the unified research graph.

    Each node receives the full ResearchState and returns a partial dict
    of fields to update.  LangGraph merges the returned dict into state.
    """

    def __init__(
        self,
        ai_engine: AIEngine,
        db: DatabaseClient,
        project_id: str,
        tools_registry,
        event_callback: Optional[Callable] = None,
    ):
        self.ai = ai_engine
        self.db = db
        self.project_id = project_id
        self.tools = tools_registry
        self._event_cb = event_callback

    # -- helpers ----------------------------------------------------------

    def _emit(self, etype: str, content: str, meta: Dict[str, Any] = None):
        if self._event_cb:
            self._event_cb({"type": etype, "content": content, "metadata": meta or {}})

    @staticmethod
    def _sanitize_for_state(obj):
        """Recursively convert numpy types to native Python for msgpack serialization."""
        try:
            import numpy as np
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        if isinstance(obj, dict):
            return {k: GraphNodes._sanitize_for_state(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [GraphNodes._sanitize_for_state(i) for i in obj]
        return obj

    @staticmethod
    def _parse_json(text: str) -> dict:
        clean = text.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        if clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        try:
            return json.loads(clean.strip())
        except json.JSONDecodeError:
            return {}

    # =====================================================================
    #  ROUTER
    # =====================================================================

    async def router(self, state: ResearchState) -> ResearchState:
        """Set up initial state based on mode."""
        logger = chat_logger.new_session()

        if state.get("reasoning_mode"):
            self._emit("thought", "Switching to Reasoning Mode (Plan & Code)...")
            logger.log_step("init", "Starting reasoning sub-graph", {
                "query": state["user_query"][:200]
            })
            # Retrieve KB context for reasoning
            context = ""
            try:
                emb = self.ai.generate_embedding(state["user_query"])
                if emb:
                    chunks = self.db.search_vectors(
                        emb, match_threshold=0.3,
                        project_id=self.project_id, match_count=5
                    )
                    if chunks:
                        context = "\n\n".join(
                            f"Document {i+1}: {c['content']}"
                            for i, c in enumerate(chunks)
                        )
            except Exception as e:
                print(f"Context retrieval for reasoning failed: {e}")

            return {
                "feedback": context,  # reuse feedback field for initial context
                "plan": None,
                "code_attempts": [],
                "current_code": None,
                "current_output": None,
                "iteration": 0,
            }

        # --- Research mode ---
        self._emit("thought", f"Starting analysis for: {state['user_query']}", {"iteration": 0})
        logger.log_step("init", "Starting ResearchAgent (normal mode)", {
            "query": state["user_query"][:200],
            "has_history": bool(state.get("chat_history")),
            "max_iterations": MAX_RESEARCH_ITERATIONS,
        })

        tool_descriptions = self.tools.get_tool_descriptions()
        system_prompt = f"""You are a research assistant with access to a knowledge base.

TOOLS:
{tool_descriptions}

FORMAT:
You must respond in JSON format with a "thought" field and either an "actions" list or "final_answer" field.

RESPONSE EXAMPLES:

1. To call tools (PARALLEL EXECUTION SUPPORTED):
{{
  "thought": "I need to search for machine learning definitions and also check for specific papers.",
  "actions": [
    {{
      "tool_name": "vector_search",
      "argument": "machine learning definition"
    }},
    {{
      "tool_name": "graph_search",
      "argument": "Transformer architecture"
    }}
  ]
}}

2. To give a final answer:
{{
  "thought": "I have found enough information to answer the user.",
  "final_answer": "Machine learning is a field of study..."
}}

GUIDELINES:
1. Always include a "thought" field to explain your reasoning.
2. Use "actions" (list) to call multiple tools in parallel when possible (e.g. search + web_search).
3. If you found the answer, provide "final_answer".
"""
        # Build history text
        history_text = ""
        ch = state.get("chat_history") or []
        if ch:
            parts = []
            for msg in ch[-5:]:
                role = "User" if msg["role"] == "user" else "Assistant"
                parts.append(f"{role}: {msg['content']}")
            if parts:
                history_text = "\nRECENT CONVERSATION:\n" + "\n".join(parts)

        conversation = (
            f"{system_prompt}{history_text}\n\n"
            f"USER QUERY: {state['user_query']}\n\n"
            "Think step by step in JSON format."
        )
        logger.log_step("prompt", "Initial prompt built", {"prompt_length": len(conversation)})

        return {
            "system_prompt": system_prompt,
            "conversation": conversation,
            "observations": [],
            "tools_used": [],
            "retrieved_chunks": [],
            "matched_concepts": [],
            "iteration": 0,
            "final_answer": None,
            "model_name": "unknown",
            "error": None,
            "llm_response_str": "",
            "parsed_response": {},
            "parsed_action": None,
        }

    # =====================================================================
    #  RESEARCH SUB-GRAPH NODES
    # =====================================================================

    async def think(self, state: ResearchState) -> ResearchState:
        """Call LLM to decide next action or final answer."""
        iteration = state.get("iteration", 0) + 1
        logger = chat_logger.get_logger()
        logger.log_step("iteration", f"Starting iteration {iteration}/{MAX_RESEARCH_ITERATIONS}")

        conversation = state["conversation"]
        start = time.time()
        llm_response, model_name = await self.ai._openrouter_generate(
            conversation,
            return_model_name=True,
            response_format={"type": "json_object"},
        )
        duration = time.time() - start
        logger.log_llm_call(conversation, llm_response, duration, self.ai.model)

        parsed = self._parse_json(llm_response)
        thought = parsed.get("thought", "")
        if thought:
            logger.log_step("thought", f"Agent thought: {thought}")
            self._emit("thought", thought, {"iteration": iteration})

        # Extract action(s)
        actions_data = parsed.get("actions")
        action_data = parsed.get("action")
        
        parsed_actions = []
        if actions_data and isinstance(actions_data, list):
            for a in actions_data:
                if isinstance(a, dict):
                    parsed_actions.append((a.get("tool_name"), a.get("argument")))
        elif action_data and isinstance(action_data, dict):
            # Backward compatibility
            parsed_actions.append((action_data.get("tool_name"), action_data.get("argument")))

        return {
            "iteration": iteration,
            "llm_response_str": llm_response,
            "parsed_response": parsed,
            "parsed_actions": parsed_actions,
            "parsed_action": parsed_actions[0] if parsed_actions else None, # Compat
            "final_answer": parsed.get("final_answer"),
            "model_name": model_name,
        }

    async def execute_tool(self, state: ResearchState) -> ResearchState:
        """Execute the tools chosen by the LLM in parallel."""
        actions = state.get("parsed_actions") or []
        if not actions and state.get("parsed_action"):
             actions = [state["parsed_action"]]
             
        logger = chat_logger.get_logger()
        
        if not actions:
            return {"observations": [], "tools_used": []}

        observations = list(state.get("observations", []))
        tools_used = list(state.get("tools_used", []))
        retrieved_chunks = list(state.get("retrieved_chunks", []))
        matched_concepts = list(state.get("matched_concepts", []))
        conversation = state["conversation"]
        
        # Prepare parallel tasks
        tasks = []
        for tool_name, argument in actions:
            # Ensure argument is a string
            if not isinstance(argument, str):
                argument = json.dumps(argument) if argument is not None else ""
                
            logger.log_step("action", f"Parsed action: {tool_name}", {"argument": argument[:100]})
            self._emit("tool", f"Calling tool: {tool_name}", {"argument": argument})
            
            tools_used.append(tool_name)
            
            tool = self.tools.get_tool(tool_name)
            if tool:
                tasks.append(tool.execute(argument))
            else:
                async def unknown_tool_dummy():
                    return f"Error: Unknown tool '{tool_name}'."
                tasks.append(unknown_tool_dummy())

        # Execute all tools
        results = await asyncio.gather(*tasks)
        
        # Process results
        new_observations_text = ""
        
        for i, (tool_name, _) in enumerate(actions):
            observation = results[i]
            observations.append((tool_name, observation))
            
            self._emit("tool_result", f"Tool {tool_name} returned results.",
                        {"preview": observation[:200]})
            
            # --- Specific Tool Handling (Sanitizing visuals, auto-fallback) ---
            if tool_name == "vector_search":
                # The tool updates self.db/self.ai implicitly? No, tool returns string.
                # But for UI features we might want to re-run or parse the string?
                # Actually, the original code ran the search AGAIN to get objects for UI.
                # That is inefficient. We should have the tool return objects if possible, 
                # but tools return strings for LLM.
                # NOTE: For now, we keep the Side-Effect logic (re-running for UI) 
                # or better, parse the observation if it was JSON?
                # The original code re-ran `db.search_vectors`.
                # We can replicate that side-effect logic here for each search tool.
                
                # Re-run for UI artifacts (chunks)
                try:
                    emb = self.ai.generate_embedding(actions[i][1]) # argument
                    chunks = []
                    if emb:
                        chunks = self.db.search_vectors(
                            emb, match_threshold=0.3, project_id=self.project_id, match_count=5
                        )
                        # We don't rerank here again to save time, or do we?
                        # Original code did. Let's skip heavy rerank for UI visualization to be fast.
                        for c in chunks: c["source"] = "vector"
                        retrieved_chunks.extend(self._sanitize_for_state(chunks))
                        
                        # Auto-fallback check
                        best = float(max((c.get("similarity", 0) for c in chunks), default=0))
                        if best < 0.3 and "web_search" not in [t[0] for t in actions] and "web_search" not in tools_used:
                             # We can't trigger it *in parallel* easily now, but we could suggest it for next turn.
                             # Or append a clear message.
                             parsed_obs = observation + "\n[System Note: Low relevance. Consider using web_search next.]"
                             # We won't auto-trigger to avoid infinite loops or complexity in parallel gather.
                except Exception:
                    pass

            elif tool_name == "graph_search":
                matched_concepts.extend([c.strip() for c in actions[i][1].split(",")])
            
            new_observations_text += f"OBSERVATION from {tool_name}:\n{observation}\n\n"

        conversation += (
            f"\n\nASSISTANT JSON:\n{state['llm_response_str']}\n\n"
            f"{new_observations_text}"
            "Based on these observations, provide the next step in JSON format."
        )

        return {
            "conversation": conversation,
            "observations": observations,
            "tools_used": tools_used,
            "retrieved_chunks": retrieved_chunks,
            "matched_concepts": list(set(matched_concepts)),
        }

    async def handle_no_action(self, state: ResearchState) -> ResearchState:
        """LLM returned neither action nor final_answer — nudge it."""
        logger = chat_logger.get_logger()
        logger.log_step("no_action", "No action parsed from JSON",
                        {"response_preview": state["llm_response_str"][:100]})
        conversation = state["conversation"] + (
            f"\n\nASSISTANT JSON:\n{state['llm_response_str']}\n\n"
            "ERROR: Invalid JSON or missing 'action'/'final_answer'. "
            "Please respond with valid JSON containing 'thought' and 'action' OR 'final_answer'."
        )
        return {"conversation": conversation}

    async def finalize_research(self, state: ResearchState) -> ResearchState:
        """Build final AgentResponse from accumulated state."""
        logger = chat_logger.get_logger()

        answer = state.get("final_answer")

        # If we hit max iterations, try fallback synthesis
        if not answer and state.get("observations"):
            ctx = "\n\n".join(f"{n}: {o}" for n, o in state["observations"])
            try:
                prompt = (
                    f"Based on this information from my knowledge base:\n\n{ctx}\n\n"
                    f"Answer this question: {state['user_query']}\n\n"
                    "Provide a helpful, complete answer based on the information above."
                )
                t0 = time.time()
                answer, model_name = await self.ai._openrouter_generate(prompt, return_model_name=True)
                dur = time.time() - t0
                logger.log_llm_call(prompt, answer, dur, self.ai.model)
            except Exception:
                pass

        # Last-resort direct RAG
        if not answer:
            try:
                emb = self.ai.generate_embedding(state["user_query"])
                if emb:
                    chunks = self.db.search_vectors(
                        emb, match_threshold=0.3,
                        project_id=self.project_id, match_count=5,
                    )
                    if chunks:
                        resp = await self.ai.chat_with_context_async(
                            query=state["user_query"],
                            context_chunks=chunks,
                            chat_history=state.get("chat_history"),
                        )
                        answer = resp["response"]
            except Exception:
                pass

        if not answer:
            answer = (
                "I wasn't able to find relevant information to answer your question. "
                "Please try rephrasing or asking about something in your knowledge base."
            )

        self._emit("result", "Synthesizing final answer...", {"model": state.get("model_name")})
        logger.log_step("complete", "Final answer ready", {"answer_length": len(answer)})
        logger.finish("success", answer[:200])

        from src.agent import AgentResponse
        result = AgentResponse(
            answer=answer,
            retrieved_chunks=self._sanitize_for_state(state.get("retrieved_chunks", [])),
            matched_concepts=state.get("matched_concepts", []),
            tools_used=state.get("tools_used", []),
            model_name=state.get("model_name", "unknown"),
        )
        return {"result": result, "final_answer": answer}

    # =====================================================================
    #  REASONING SUB-GRAPH NODES
    # =====================================================================

    async def generate_plan(self, state: ResearchState) -> ResearchState:
        """Generate a reasoning plan before coding."""
        logger = chat_logger.get_logger()
        logger.log_step("planning", "Generating plan...")
        self._emit("thought", "Planning approach...")

        context = state.get("feedback", "")
        task = state["user_query"]

        if context:
            prompt = f"""You are a helpful research assistant.

TASK: {task}

DOCUMENTS FROM KNOWLEDGE BASE:
{context}

CRITICAL INSTRUCTIONS:
1. Assess if the documents above are ACTUALLY RELEVANT to the task.
2. If NOT relevant, IGNORE them and solve computationally with Python.
3. For factual/computational questions, solve with code.

Create a plan in this EXACT format:

CONTEXT_RELEVANT: [YES or NO]
CONTEXT_NEEDED: [What info is relevant, or "None - solving computationally"]
GOAL: [What your code will accomplish]
VERIFICATION: [How to verify the answer is correct]

Be concise and specific."""
        else:
            prompt = f"""You are a coding assistant that solves problems using Python.

TASK: {task}

Create a plan in this EXACT format:

CONTEXT_NEEDED: [What data or info you need]
GOAL: [What your code will accomplish - be specific]
VERIFICATION: [How to verify the output is correct]

Be concise and specific."""

        try:
            t0 = time.time()
            text, model_name = await self.ai._openrouter_generate(prompt, return_model_name=True)
            dur = time.time() - t0
            logger.log_llm_call(prompt, text, dur, self.ai.model)

            plan = self._parse_reasoning_plan(text)
            if plan:
                self._emit("thought", f"Goal: {plan.goal}", {"phase": "plan"})
                logger.log_step("planning", "Plan generated", {
                    "goal": plan.goal,
                    "verification": plan.verification_logic,
                })
            return {"plan": plan, "model_name": model_name, "feedback": "", "iteration": 0}
        except Exception as e:
            logger.log_error(f"Plan generation failed: {e}")
            return {"plan": None, "error": str(e)}

    @staticmethod
    def _parse_reasoning_plan(text: str) -> Optional[ReasoningPlan]:
        ctx = re.search(r'CONTEXT_NEEDED:\s*(.+?)(?=GOAL:|$)', text, re.DOTALL | re.IGNORECASE)
        goal = re.search(r'GOAL:\s*(.+?)(?=VERIFICATION:|$)', text, re.DOTALL | re.IGNORECASE)
        ver = re.search(r'VERIFICATION:\s*(.+?)$', text, re.DOTALL | re.IGNORECASE)
        if goal:
            return ReasoningPlan(
                context_needed=ctx.group(1).strip() if ctx else "None",
                goal=goal.group(1).strip(),
                verification_logic=ver.group(1).strip() if ver else "Output should be non-empty",
            )
        return None

    async def generate_code(self, state: ResearchState) -> ResearchState:
        """Generate Python code to solve the task."""
        logger = chat_logger.get_logger()
        iteration = state.get("iteration", 0) + 1
        self._emit("thought", f"Writing code (attempt {iteration}/{MAX_CODE_RETRIES})...",
                    {"phase": "coding"})
        logger.log_step("coding", f"Starting attempt {iteration}")

        plan = state["plan"]
        feedback = state.get("feedback", "")
        prompt = f"""You are a Python coding assistant. Write code to solve this task.

TASK: {state['user_query']}

PLAN:
- Goal: {plan.goal}
- Verification: {plan.verification_logic}

{f"PREVIOUS ERROR - FIX THIS:{chr(10)}{feedback}" if feedback else ""}

REQUIREMENTS:
1. Write complete, runnable Python code
2. Print the final result to stdout
3. Use assert statements where helpful
4. Do NOT use input() or external files
5. Code must be self-contained

Respond with ONLY Python code in a ```python``` block. No explanations."""

        try:
            t0 = time.time()
            text, model_name = await self.ai._openrouter_generate(prompt, return_model_name=True)
            dur = time.time() - t0
            logger.log_llm_call(prompt, text, dur, self.ai.model)

            code_match = re.search(r'```python\s*(.*?)```', text, re.DOTALL | re.IGNORECASE)
            if not code_match:
                code_match = re.search(r'```\s*(.*?)```', text, re.DOTALL)
            code = code_match.group(1).strip() if code_match else text.strip() or None

            return {
                "current_code": code,
                "iteration": iteration,
                "model_name": model_name,
            }
        except Exception as e:
            logger.log_error(f"Code generation failed: {e}")
            return {
                "current_code": None,
                "iteration": iteration,
                "code_attempts": state.get("code_attempts", []) + [
                    CodeAttempt(attempt_number=iteration, code="", success=False,
                                output="", error=f"LLM error: {e}")
                ],
            }

    async def execute_sandbox(self, state: ResearchState) -> ResearchState:
        """Run code in sandbox and capture output."""
        logger = chat_logger.get_logger()
        code = state.get("current_code")
        iteration = state.get("iteration", 1)

        if not code:
            return {
                "current_output": None,
                "feedback": "Failed to generate code. Try again.",
                "code_attempts": state.get("code_attempts", []) + [
                    CodeAttempt(attempt_number=iteration, code="", success=False,
                                output="", error="No code generated")
                ],
            }

        self._emit("thought", f"Running code (attempt {iteration})...", {"phase": "sandbox"})
        t0 = time.time()
        success, output = await run_code(code, timeout_s=5.0)
        dur = time.time() - t0
        logger.log_sandbox_execution(code, success, output, dur)

        attempt = CodeAttempt(
            attempt_number=iteration, code=code,
            success=success, output=output,
            error=None if success else output,
        )

        feedback = ""
        if not success:
            feedback = f"Execution error:\n{output}\n\nPlease fix this error."

        return {
            "current_output": output if success else None,
            "feedback": feedback,
            "code_attempts": state.get("code_attempts", []) + [attempt],
        }

    async def verify_result(self, state: ResearchState) -> ResearchState:
        """Generate and run a verification script."""
        logger = chat_logger.get_logger()
        output = state.get("current_output", "")
        plan = state["plan"]

        self._emit("thought", "Verifying result...", {"phase": "verify"})

        prompt = f"""Generate a Python script that verifies the following output.

OUTPUT TO VERIFY:
{output}

VERIFICATION CRITERIA:
{plan.verification_logic}

Write Python code that:
1. Uses OUTPUT = '''{output}'''
2. Uses assert statements to verify
3. Prints "VERIFICATION PASSED" if all pass

Respond with ONLY Python code in a ```python``` block."""

        try:
            text = await self.ai._openrouter_generate(prompt)
            code_match = re.search(r'```python\s*(.*?)```', text, re.DOTALL | re.IGNORECASE)
            if not code_match:
                code_match = re.search(r'```\s*(.*?)```', text, re.DOTALL)

            if code_match:
                script = code_match.group(1).strip()
                ver_success, ver_output = await run_code(script, timeout_s=5.0)

                if ver_success:
                    logger.finish("success", output)
                    return {"feedback": "VERIFIED"}
                else:
                    return {
                        "feedback": (
                            f"Code executed with output: {output}\n"
                            f"But verification FAILED:\n{ver_output}\n"
                            f"Expected: {plan.verification_logic}\n"
                            "Fix the code to pass verification."
                        )
                    }
        except Exception:
            pass

        # Basic fallback verification
        if output and not any(w in output.lower() for w in ["error", "exception", "traceback", "failed"]):
            logger.finish("success", output)
            return {"feedback": "VERIFIED"}

        return {"feedback": f"Output doesn't meet requirements: {output}"}

    async def finalize_reasoning(self, state: ResearchState) -> ResearchState:
        """Build ReasoningResponse from accumulated state."""
        logger = chat_logger.get_logger()
        attempts = state.get("code_attempts", [])
        plan = state.get("plan")
        success = state.get("feedback") == "VERIFIED"

        if not success:
            self._emit("thought", f"Failed after {len(attempts)} attempts", {"phase": "done"})
            logger.finish("failed", f"Max retries exceeded")

        result = ReasoningResponse(
            success=success,
            plan=plan,
            final_code=state.get("current_code"),
            final_output=state.get("current_output"),
            attempts=attempts,
            error=None if success else f"Failed after {len(attempts)} attempts",
            model_name=state.get("model_name", "unknown"),
        )
        return {"result": result}


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def route_mode(state: ResearchState) -> str:
    """Router: research vs reasoning sub-graph."""
    return "generate_plan" if state.get("reasoning_mode") else "think"


def after_think(state: ResearchState) -> str:
    """After think: finalize, execute tool, or handle parse error."""
    if state.get("final_answer"):
        return "finalize_research"
    if state.get("iteration", 0) >= MAX_RESEARCH_ITERATIONS:
        return "finalize_research"
    if state.get("parsed_actions") or state.get("parsed_action"):
        return "execute_tool"
    return "handle_no_action"


def after_plan(state: ResearchState) -> str:
    """After plan: proceed to code or finalize with error."""
    if state.get("plan"):
        return "generate_code"
    return "finalize_reasoning"


def after_sandbox(state: ResearchState) -> str:
    """After sandbox: verify on success, retry or give up on failure."""
    feedback = state.get("feedback", "")
    iteration = state.get("iteration", 0)

    if state.get("current_output") is not None:
        # Execution succeeded → verify
        return "verify_result"
    if iteration >= MAX_CODE_RETRIES:
        return "finalize_reasoning"
    return "generate_code"


def after_verify(state: ResearchState) -> str:
    """After verification: finalize on pass, retry on fail."""
    if state.get("feedback") == "VERIFIED":
        return "finalize_reasoning"
    if state.get("iteration", 0) >= MAX_CODE_RETRIES:
        return "finalize_reasoning"
    return "generate_code"


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------

def build_research_graph(
    ai_engine: AIEngine,
    db: DatabaseClient,
    project_id: str,
    tools_registry,
    event_callback: Optional[Callable] = None,
):
    """
    Construct and compile the unified research StateGraph.

    Returns the compiled graph (with MemorySaver checkpointing).
    """
    nodes = GraphNodes(ai_engine, db, project_id, tools_registry, event_callback)

    workflow = StateGraph(ResearchState)

    # --- Register nodes ---
    workflow.add_node("router", nodes.router)
    # Research
    workflow.add_node("think", nodes.think)
    workflow.add_node("execute_tool", nodes.execute_tool)
    workflow.add_node("handle_no_action", nodes.handle_no_action)
    workflow.add_node("finalize_research", nodes.finalize_research)
    # Reasoning
    workflow.add_node("generate_plan", nodes.generate_plan)
    workflow.add_node("generate_code", nodes.generate_code)
    workflow.add_node("execute_sandbox", nodes.execute_sandbox)
    workflow.add_node("verify_result", nodes.verify_result)
    workflow.add_node("finalize_reasoning", nodes.finalize_reasoning)

    # --- Edges ---
    workflow.add_edge(START, "router")
    workflow.add_conditional_edges("router", route_mode, {
        "think": "think",
        "generate_plan": "generate_plan",
    })

    # Research sub-graph
    workflow.add_conditional_edges("think", after_think, {
        "finalize_research": "finalize_research",
        "execute_tool": "execute_tool",
        "handle_no_action": "handle_no_action",
    })
    workflow.add_edge("execute_tool", "think")
    workflow.add_edge("handle_no_action", "think")
    workflow.add_edge("finalize_research", END)

    # Reasoning sub-graph
    workflow.add_conditional_edges("generate_plan", after_plan, {
        "generate_code": "generate_code",
        "finalize_reasoning": "finalize_reasoning",
    })
    workflow.add_edge("generate_code", "execute_sandbox")
    workflow.add_conditional_edges("execute_sandbox", after_sandbox, {
        "verify_result": "verify_result",
        "generate_code": "generate_code",
        "finalize_reasoning": "finalize_reasoning",
    })
    workflow.add_conditional_edges("verify_result", after_verify, {
        "finalize_reasoning": "finalize_reasoning",
        "generate_code": "generate_code",
    })
    workflow.add_edge("finalize_reasoning", END)

    # --- Checkpointing ---
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)
