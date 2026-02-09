"""
Chat debugging logger for the Research Assistant.

Creates detailed log files in logs/ directory for each chat session,
capturing LLM prompts, responses, and timing information.
"""

import os
import json
import time
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Log directory
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


class ChatLogger:
    """
    Logs all steps of a chat session for debugging.
    
    Usage:
        logger = ChatLogger()
        logger.log_step("planning", "Generating plan", {"prompt": "..."})
        logger.log_llm_call(prompt, response, duration)
        logger.save()
    """
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = LOG_DIR / f"chat_{self.session_id}.json"
        self.start_time = time.time()
        self.steps: list = []
        self.metadata = {
            "session_id": self.session_id,
            "started_at": datetime.now().isoformat(),
            "ended_at": None,
            "total_duration_s": None,
            "status": "in_progress"
        }
        
    def log_step(self, phase: str, message: str, data: Optional[dict] = None):
        """Log a general step in the process."""
        step = {
            "type": "step",
            "timestamp": datetime.now().isoformat(),
            "elapsed_s": round(time.time() - self.start_time, 2),
            "phase": phase,
            "message": message,
            "data": data or {}
        }
        self.steps.append(step)
        self._write_to_file()
        print(f"[LOG {phase}] {message}")
        
    def log_llm_call(self, prompt: str, response: str, duration_s: float, model: str = "unknown"):
        """Log an LLM API call with timing."""
        step = {
            "type": "llm_call",
            "timestamp": datetime.now().isoformat(),
            "elapsed_s": round(time.time() - self.start_time, 2),
            "model": model,
            "duration_s": round(duration_s, 2),
            "prompt_length": len(prompt),
            "response_length": len(response),
            "prompt": prompt[:2000] + "..." if len(prompt) > 2000 else prompt,  # Truncate for readability
            "response": response[:2000] + "..." if len(response) > 2000 else response
        }
        self.steps.append(step)
        self._write_to_file()
        print(f"[LOG LLM] Model={model}, Duration={duration_s:.2f}s, Response={response[:100]}...")
        
    def log_sandbox_execution(self, code: str, success: bool, output: str, duration_s: float):
        """Log a sandbox code execution."""
        step = {
            "type": "sandbox",
            "timestamp": datetime.now().isoformat(),
            "elapsed_s": round(time.time() - self.start_time, 2),
            "success": success,
            "duration_s": round(duration_s, 2),
            "code": code[:1000] + "..." if len(code) > 1000 else code,
            "output": output[:500] + "..." if len(output) > 500 else output
        }
        self.steps.append(step)
        self._write_to_file()
        print(f"[LOG SANDBOX] Success={success}, Duration={duration_s:.2f}s")
        
    def log_error(self, error: str, context: Optional[dict] = None):
        """Log an error."""
        step = {
            "type": "error",
            "timestamp": datetime.now().isoformat(),
            "elapsed_s": round(time.time() - self.start_time, 2),
            "error": str(error),
            "context": context or {}
        }
        self.steps.append(step)
        self._write_to_file()
        print(f"[LOG ERROR] {error}")
        
    def finish(self, status: str = "completed", result: Optional[str] = None):
        """Mark the session as finished."""
        self.metadata["ended_at"] = datetime.now().isoformat()
        self.metadata["total_duration_s"] = round(time.time() - self.start_time, 2)
        self.metadata["status"] = status
        if result:
            self.metadata["result_preview"] = result[:500]
        self._write_to_file()
        print(f"[LOG FINISH] Status={status}, Duration={self.metadata['total_duration_s']}s")
        
    def _write_to_file(self):
        """Write current state to log file."""
        try:
            log_data = {
                "metadata": self.metadata,
                "steps": self.steps
            }
            with open(self.log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            print(f"[LOG] Failed to write log file: {e}")


# REQ-ASYNC-01: Use contextvars for async-safe logger instance (replaces global mutable state)
_current_logger: ContextVar[Optional[ChatLogger]] = ContextVar('current_logger', default=None)


def get_logger() -> ChatLogger:
    """Get or create the current chat logger (async-safe via contextvars)."""
    logger = _current_logger.get()
    if logger is None:
        logger = ChatLogger()
        _current_logger.set(logger)
    return logger


def new_session(session_id: Optional[str] = None) -> ChatLogger:
    """Start a new logging session (async-safe via contextvars)."""
    logger = ChatLogger(session_id)
    _current_logger.set(logger)
    return logger


def log_step(phase: str, message: str, data: Optional[dict] = None):
    """Convenience function to log a step."""
    get_logger().log_step(phase, message, data)


def log_llm_call(prompt: str, response: str, duration_s: float, model: str = "unknown"):
    """Convenience function to log an LLM call."""
    get_logger().log_llm_call(prompt, response, duration_s, model)


def log_error(error: str, context: Optional[dict] = None):
    """Convenience function to log an error."""
    get_logger().log_error(error, context)


def finish_session(status: str = "completed", result: Optional[str] = None):
    """Convenience function to finish the session."""
    get_logger().finish(status, result)

