"""
REQ-POETIQ-01: Python Sandbox Tool

Secure execution environment for Python code, adapted from Poetiq's arc_agi/sandbox.py.
Runs code in an isolated subprocess with timeout and restricted environment.
"""

import asyncio
import os
import sys
import tempfile
import textwrap
from typing import Tuple


async def run_code(
    code: str, 
    timeout_s: float = 5.0
) -> Tuple[bool, str]:
    """
    Execute Python code in an isolated subprocess.
    
    Args:
        code: Python code to execute
        timeout_s: Maximum execution time in seconds
        
    Returns:
        Tuple of (success: bool, output_or_error: str)
        - On success: (True, stdout output)
        - On failure: (False, error message)
    """
    # Wrap the code in a script that captures output
    script = _build_script(code)
    
    with tempfile.TemporaryDirectory() as td:
        script_path = os.path.join(td, "user_code.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)
        
        # Create restricted environment
        env = {
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PATH": os.environ.get("PATH", ""),
        }
        
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=td,
            env=env,
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
            return False, f"Execution timeout after {timeout_s}s"
        
        # Check for errors
        if proc.returncode != 0:
            error_output = stderr.decode().strip() or stdout.decode().strip()
            return False, error_output
        
        # Return stdout on success
        output = stdout.decode().strip()
        return True, output


def _build_script(code: str) -> str:
    """
    Build a complete Python script from user code.
    
    The script captures all output and handles common imports.
    """
    return textwrap.dedent(f"""\
# User code execution wrapper
import sys
import traceback

# Common imports available to user code
import json
import math
import re
import collections
import itertools
import functools
from typing import List, Dict, Any, Optional, Tuple

try:
{textwrap.indent(code, '    ')}
except Exception as e:
    traceback.print_exc()
    sys.exit(1)
""")


def run_code_sync(code: str, timeout_s: float = 5.0) -> Tuple[bool, str]:
    """
    Synchronous wrapper for run_code.
    
    Useful for non-async contexts.
    """
    return asyncio.run(run_code(code, timeout_s))
