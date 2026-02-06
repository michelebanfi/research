"""
REQ-POETIQ-02, REQ-POETIQ-03, REQ-POETIQ-04, REQ-POETIQ-05: Reasoning Agent

Implements the "Plan & Code" loop with:
- Goal & Plan phase before coding
- Iterative coding loop with error feedback
- Self-verification with assertions
"""

import re
from typing import Optional, List, Dict, Any, Callable

from src.models import ReasoningPlan, CodeAttempt, ReasoningResponse
from src.sandbox import run_code


class ReasoningAgent:
    """
    Plan & Code agent with iterative error correction.
    
    REQ-POETIQ-02: When reasoning_mode=True, uses this specialized loop
    instead of standard ReAct.
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
        
        # Phase 1: Goal & Plan (REQ-POETIQ-03)
        self._notify("planning", "Generating plan...")
        plan = await self._generate_plan(task, context)
        
        if not plan:
            return ReasoningResponse(
                success=False,
                error="Failed to generate a plan for this task."
            )
        
        self._notify("planning", f"Goal: {plan.goal}")
        
        # Phase 2: Iterative Coding Loop (REQ-POETIQ-04)
        feedback = ""
        
        for attempt_num in range(1, self.MAX_CODE_RETRIES + 1):
            self._notify("coding", f"Attempt {attempt_num}/{self.MAX_CODE_RETRIES}")
            
            # Generate code
            code = await self._generate_code(task, plan, feedback)
            
            if not code:
                attempts.append(CodeAttempt(
                    attempt_number=attempt_num,
                    code="",
                    success=False,
                    output="",
                    error="Failed to generate code"
                ))
                continue
            
            # Execute in sandbox
            self._notify("executing", f"Running code (attempt {attempt_num})...")
            success, output = await run_code(code, timeout_s=5.0)
            
            attempt = CodeAttempt(
                attempt_number=attempt_num,
                code=code,
                success=success,
                output=output,
                error=None if success else output
            )
            attempts.append(attempt)
            
            if success:
                # Phase 3: Self-Verification (REQ-POETIQ-05)
                self._notify("verifying", "Verifying output...")
                
                if self._verify_output(output, plan.verification_logic):
                    self._notify("complete", "Task completed successfully!")
                    return ReasoningResponse(
                        success=True,
                        plan=plan,
                        final_code=code,
                        final_output=output,
                        attempts=attempts
                    )
                else:
                    # Verification failed - try again
                    feedback = f"Code executed but verification failed. Output: {output}\nExpected: {plan.verification_logic}\nPlease fix the code."
            else:
                # Execution failed - feed error back
                feedback = f"Execution error:\n{output}\n\nPlease fix this error."
        
        # Max retries reached
        self._notify("failed", f"Failed after {self.MAX_CODE_RETRIES} attempts")
        
        return ReasoningResponse(
            success=False,
            plan=plan,
            final_code=attempts[-1].code if attempts else None,
            final_output=attempts[-1].output if attempts else None,
            attempts=attempts,
            error=f"Failed to solve task after {self.MAX_CODE_RETRIES} attempts"
        )
    
    async def _generate_plan(self, task: str, context: str) -> Optional[ReasoningPlan]:
        """
        REQ-POETIQ-03: Generate a plan before coding.
        
        Prompts the LLM to define:
        1. Context needed
        2. Goal
        3. Verification logic
        """
        prompt = f"""You are a coding assistant that solves problems using Python.

TASK: {task}

{f"CONTEXT: {context}" if context else ""}

Before writing any code, you must create a plan. Respond in this EXACT format:

CONTEXT_NEEDED: [What data or information you need to solve this task]
GOAL: [What your code will accomplish - be specific]
VERIFICATION: [How to verify the output is correct, e.g., "Output should be a number greater than 0"]

Be concise and specific."""

        try:
            response = await self.ai.async_client.generate(
                model=self.ai.model,
                prompt=prompt
            )
            text = response.get("response", "")
            
            return self._parse_plan(text)
            
        except Exception as e:
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
    ) -> Optional[str]:
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
            response = await self.ai.async_client.generate(
                model=self.ai.model,
                prompt=prompt
            )
            text = response.get("response", "")
            
            # Extract code from markdown block
            code_match = re.search(r'```python\s*(.*?)```', text, re.DOTALL | re.IGNORECASE)
            if code_match:
                return code_match.group(1).strip()
            
            # Try without markdown
            code_match = re.search(r'```\s*(.*?)```', text, re.DOTALL)
            if code_match:
                return code_match.group(1).strip()
            
            # Return raw text if no code block
            return text.strip() if text.strip() else None
            
        except Exception as e:
            return None
    
    def _verify_output(self, output: str, verification_logic: str) -> bool:
        """
        REQ-POETIQ-05: Verify the output meets the verification criteria.
        
        Simple verification - checks if output exists.
        More sophisticated checks could be added based on verification_logic.
        """
        if not output:
            return False
        
        # Basic verification: output exists and doesn't contain error indicators
        error_indicators = ["error", "exception", "traceback", "failed"]
        output_lower = output.lower()
        
        for indicator in error_indicators:
            if indicator in output_lower:
                # Check if it's part of the verification logic (expected)
                if indicator not in verification_logic.lower():
                    return False
        
        return True
