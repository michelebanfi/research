"""
REQ-POETIQ-02, REQ-POETIQ-03, REQ-POETIQ-04, REQ-POETIQ-05, REQ-POETIQ-06: Reasoning Agent

Implements the "Plan & Code" loop with:
- Goal & Plan phase before coding
- Iterative coding loop with error feedback
- Self-verification with executable assertions (REQ-POETIQ-06)
"""

import re
import time
from typing import Optional, List, Dict, Any, Callable

from src.models import ReasoningPlan, CodeAttempt, ReasoningResponse
from src.sandbox import run_code
from src import chat_logger


class ReasoningAgent:
    """
    Plan & Code agent with iterative error correction.
    
    REQ-POETIQ-02: When reasoning_mode=True, uses this specialized loop
    instead of standard ReAct.
    
    REQ-POETIQ-06: Uses executable Python assertion scripts for verification.
    """
    
    MAX_CODE_RETRIES = 5  # REQ-POETIQ-04
    
    def __init__(
        self,
        ai_engine,
        status_callback: Optional[Callable[[str, str], None]] = None
    ):
        """
        Initialize the Reasoning Agent.
        
        Args:
            ai_engine: AIEngine instance for LLM calls
            status_callback: Optional callback(phase, message) for UI updates
        """
        self.ai = ai_engine
        self.status_callback = status_callback
    
    def _notify(self, phase: str, message: str):
        """Notify UI of current reasoning phase."""
        if self.status_callback:
            self.status_callback(phase, message)
    
    async def run(
        self,
        task: str,
        context: str = ""
    ) -> ReasoningResponse:
        """
        Run the Plan & Code loop to solve a task.
        
        Args:
            task: The user's task/question to solve
            context: Optional additional context
            
        Returns:
            ReasoningResponse with plan, attempts, and result
        """
        attempts: List[CodeAttempt] = []
        
        # Start logging session
        logger = chat_logger.new_session()
        logger.log_step("init", "Starting reasoning agent", {
            "task": task[:200],
            "has_context": bool(context),
            "max_retries": self.MAX_CODE_RETRIES
        })
        
        # Phase 1: Goal & Plan (REQ-POETIQ-03)
        self._notify("planning", "Generating plan...")
        logger.log_step("planning", "Generating plan...")
        
        plan, model_name = await self._generate_plan(task, context)
        
        if not plan:
            logger.log_error("Failed to generate plan")
            logger.finish("failed", "No plan generated")
            return ReasoningResponse(
                success=False,
                error="Failed to generate a plan for this task.",
                model_name=model_name
            )
        
        logger.log_step("planning", f"Plan generated", {
            "goal": plan.goal,
            "context_needed": plan.context_needed,
            "verification": plan.verification_logic
        })
        self._notify("planning", f"Goal: {plan.goal}")
        
        # Phase 2: Iterative Coding Loop (REQ-POETIQ-04)
        feedback = ""
        
        for attempt_num in range(1, self.MAX_CODE_RETRIES + 1):
            self._notify("coding", f"Attempt {attempt_num}/{self.MAX_CODE_RETRIES}")
            logger.log_step("coding", f"Starting attempt {attempt_num}")
            
            # Generate code
            code, coding_model = await self._generate_code(task, plan, feedback)
            if coding_model:
                model_name = coding_model  # Update model name to latest used
            
            if not code:
                logger.log_error(f"Failed to generate code on attempt {attempt_num}")
                attempts.append(CodeAttempt(
                    attempt_number=attempt_num,
                    code="",
                    success=False,
                    output="",
                    error="Failed to generate code"
                ))
                continue
            
            logger.log_step("coding", f"Code generated", {"code_length": len(code)})
            
            # Execute in sandbox (1st execution)
            self._notify("executing", f"Running code (attempt {attempt_num})...")
            start_time = time.time()
            success, output = await run_code(code, timeout_s=5.0)
            exec_duration = time.time() - start_time
            
            logger.log_sandbox_execution(code, success, output, exec_duration)
            
            attempt = CodeAttempt(
                attempt_number=attempt_num,
                code=code,
                success=success,
                output=output,
                error=None if success else output
            )
            
            if success:
                # REQ-POETIQ-06: Executable Verification (2nd execution)
                self._notify("verifying", "Generating verification script...")
                
                verification_script = await self._generate_verification_script(
                    output, plan.verification_logic
                )
                
                if verification_script:
                    self._notify("verifying", "Running verification assertions...")
                    ver_success, ver_output = await run_code(verification_script, timeout_s=5.0)
                    attempt.verification_output = ver_output
                    
                    if ver_success:
                        self._notify("complete", "Task completed and verified!")
                        logger.finish("success", output)
                        attempts.append(attempt)
                        return ReasoningResponse(
                            success=True,
                            plan=plan,
                            final_code=code,
                            final_output=output,
                            attempts=attempts,
                            model_name=model_name
                        )
                    else:
                        # Verification script failed (AssertionError or other error)
                        feedback = (
                            f"Code executed successfully with output: {output}\n"
                            f"But verification FAILED with error:\n{ver_output}\n"
                            f"Expected: {plan.verification_logic}\n"
                            f"Fix the code to pass verification."
                        )
                else:
                    # Couldn't generate verification script, fall back to basic check
                    if self._basic_verify_output(output):
                        self._notify("complete", "Task completed (basic verification)!")
                        logger.finish("success", output)
                        attempts.append(attempt)
                        return ReasoningResponse(
                            success=True,
                            plan=plan,
                            final_code=code,
                            final_output=output,
                            attempts=attempts,
                            model_name=model_name
                        )
                    else:
                        feedback = f"Output doesn't meet basic requirements: {output}"
            else:
                # Execution failed - feed error back
                feedback = f"Execution error:\n{output}\n\nPlease fix this error."
            
            attempts.append(attempt)
        
        # Max retries reached
        self._notify("failed", f"Failed after {self.MAX_CODE_RETRIES} attempts")
        logger.finish("failed", f"Max retries ({self.MAX_CODE_RETRIES}) exceeded")
        
        return ReasoningResponse(
            success=False,
            plan=plan,
            final_code=attempts[-1].code if attempts else None,
            final_output=attempts[-1].output if attempts else None,
            attempts=attempts,
            error=f"Failed to solve task after {self.MAX_CODE_RETRIES} attempts",
            model_name=model_name
        )
    
    async def _generate_plan(self, task: str, context: str) -> tuple[Optional[ReasoningPlan], str]:
        """
        REQ-POETIQ-03: Generate a plan before coding.
        
        If context is provided (from knowledge base), the plan focuses on analysis.
        Otherwise, it focuses on computational Python code.
        """
        if context:
            # Knowledge-based task with context from documents
            prompt = f"""You are a helpful research assistant.

TASK: {task}

DOCUMENTS FROM KNOWLEDGE BASE:
{context}

CRITICAL INSTRUCTIONS:
1. First, assess if the documents above are ACTUALLY RELEVANT to answering the task.
2. If the documents are NOT relevant (e.g., unrelated content, garbage data, or don't answer the question), 
   you MUST IGNORE them and solve the task computationally using Python code.
3. For factual/computational questions (math, algorithms, well-known facts like Fibonacci numbers),
   solve them directly with code rather than extracting from documents.

Before answering, create a plan. Respond in this EXACT format:

CONTEXT_RELEVANT: [YES if documents genuinely help answer the task, NO if they should be ignored]
CONTEXT_NEEDED: [What specific information from the documents is relevant, or "None - solving computationally"]
GOAL: [What your code will accomplish - either extracting from documents OR computing directly]
VERIFICATION: [How to verify the answer is correct, e.g., "Fibonacci(21) should equal 10946"]

Be concise and specific."""
        else:
            # Computational task with no context
            prompt = f"""You are a coding assistant that solves problems using Python.

TASK: {task}

Before writing any code, you must create a plan. Respond in this EXACT format:

CONTEXT_NEEDED: [What data or information you need to solve this task]
GOAL: [What your code will accomplish - be specific]
VERIFICATION: [How to verify the output is correct, e.g., "Output should be a number greater than 0"]

Be concise and specific."""

        try:
            start_time = time.time()
            chat_logger.log_step("llm_call", "Calling LLM for plan generation", {"prompt_length": len(prompt)})
            
            text, model_name = await self.ai._openrouter_generate(prompt, return_model_name=True)
            
            duration = time.time() - start_time
            chat_logger.log_llm_call(prompt, text, duration, self.ai.model)
            
            return self._parse_plan(text), model_name
            
        except Exception as e:
            chat_logger.log_error(f"LLM call failed in _generate_plan: {e}")
            return None
    
    def _parse_plan(self, text: str) -> Optional[ReasoningPlan]:
        """Parse the LLM response into a ReasoningPlan."""
        context_match = re.search(r'CONTEXT_NEEDED:\s*(.+?)(?=GOAL:|$)', text, re.DOTALL | re.IGNORECASE)
        goal_match = re.search(r'GOAL:\s*(.+?)(?=VERIFICATION:|$)', text, re.DOTALL | re.IGNORECASE)
        verification_match = re.search(r'VERIFICATION:\s*(.+?)$', text, re.DOTALL | re.IGNORECASE)
        
        if goal_match:
            return ReasoningPlan(
                context_needed=context_match.group(1).strip() if context_match else "None specified",
                goal=goal_match.group(1).strip(),
                verification_logic=verification_match.group(1).strip() if verification_match else "Output should be non-empty"
            )
        
        return None
    
    async def _generate_code(
        self,
        task: str,
        plan: ReasoningPlan,
        feedback: str = ""
    ) -> tuple[Optional[str], str]:
        """
        REQ-POETIQ-04, REQ-POETIQ-05: Generate code with self-verification.
        
        The prompt instructs the model to include assertions and print statements.
        """
        prompt = f"""You are a Python coding assistant. Write code to solve this task.

TASK: {task}

PLAN:
- Goal: {plan.goal}
- Verification: {plan.verification_logic}

{f"PREVIOUS ERROR - FIX THIS:{chr(10)}{feedback}" if feedback else ""}

REQUIREMENTS:
1. Write complete, runnable Python code
2. Print the final result to stdout
3. Use assert statements for intermediate checks where helpful
4. Do NOT use input() or external files
5. The code should be self-contained

Respond with ONLY Python code in a ```python``` block. No explanations."""

        try:
            start_time = time.time()
            chat_logger.log_step("llm_call", "Calling LLM for code generation", {"prompt_length": len(prompt)})
            
            text, model_name = await self.ai._openrouter_generate(prompt, return_model_name=True)
            
            duration = time.time() - start_time
            chat_logger.log_llm_call(prompt, text, duration, self.ai.model)
            
            # Extract code from markdown block
            code_match = re.search(r'```python\s*(.*?)```', text, re.DOTALL | re.IGNORECASE)
            if code_match:
                return code_match.group(1).strip(), model_name
            
            # Try without markdown
            code_match = re.search(r'```\s*(.*?)```', text, re.DOTALL)
            if code_match:
                return code_match.group(1).strip(), model_name
            
            # Return raw text if no code block
            return (text.strip() if text.strip() else None), model_name
            
        except Exception as e:
            chat_logger.log_error(f"LLM call failed in _generate_code: {e}")
            return None
    
    async def _generate_verification_script(
        self, 
        output: str, 
        verification_logic: str
    ) -> Optional[str]:
        """
        REQ-POETIQ-06: Generate a Python assertion script to verify output.
        
        The script should raise AssertionError if verification fails.
        """
        prompt = f"""Generate a Python script that verifies the following output matches the expected criteria.

OUTPUT TO VERIFY:
{output}

VERIFICATION CRITERIA:
{verification_logic}

Write Python code that:
1. Parses the output (it's provided as the string variable OUTPUT below)
2. Uses assert statements to verify it meets the criteria
3. If all assertions pass, print "VERIFICATION PASSED"
4. If any assertion fails, it will raise AssertionError automatically

Start with: OUTPUT = '''{output}'''

Respond with ONLY Python code in a ```python``` block. No explanations."""

        try:
            text = await self.ai._openrouter_generate(prompt)
            
            # Extract code from markdown block
            code_match = re.search(r'```python\s*(.*?)```', text, re.DOTALL | re.IGNORECASE)
            if code_match:
                return code_match.group(1).strip()
            
            # Try without markdown
            code_match = re.search(r'```\s*(.*?)```', text, re.DOTALL)
            if code_match:
                return code_match.group(1).strip()
            
            return None
            
        except Exception as e:
            return None
    
    def _basic_verify_output(self, output: str) -> bool:
        """
        Fallback verification when assertion script generation fails.
        
        Checks if output exists and doesn't contain obvious error indicators.
        """
        if not output:
            return False
        
        # Basic verification: output exists and doesn't contain error indicators
        error_indicators = ["error", "exception", "traceback", "failed"]
        output_lower = output.lower()
        
        for indicator in error_indicators:
            if indicator in output_lower:
                return False
        
        return True

