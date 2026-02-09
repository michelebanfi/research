"""
REQ-POETIQ-03, REQ-POETIQ-05: Data models for Reasoning Mode.

Pydantic models for structured reasoning with goal/plan phase and verification.
"""

from typing import Optional, List
from dataclasses import dataclass, field


@dataclass
class ReasoningPlan:
    """
    REQ-POETIQ-03: Goal & Plan Phase structure.
    
    Before coding, the model must explicitly state:
    1. Context needed - what data/information is required
    2. Goal - what the code should accomplish
    3. Verification logic - how to verify success
    """
    context_needed: str
    goal: str
    verification_logic: str


@dataclass
class CodeAttempt:
    """Represents a single code generation attempt."""
    attempt_number: int
    code: str
    success: bool
    output: str
    error: Optional[str] = None
    verification_output: Optional[str] = None  # REQ-POETIQ-06: Output from verification script


@dataclass
class ReasoningResponse:
    """
    Response from the reasoning agent after processing a task.
    
    Contains the plan, all code attempts, and final result.
    """
    success: bool
    plan: Optional[ReasoningPlan] = None
    final_code: Optional[str] = None
    final_output: Optional[str] = None
    attempts: List[CodeAttempt] = field(default_factory=list)
    error: Optional[str] = None
    
    @property
    def total_attempts(self) -> int:
        return len(self.attempts)
