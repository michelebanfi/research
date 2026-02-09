"""
REQ-POETIQ-01: Python Sandbox Tool
REQ-SEC-01: Static Analysis Security Scanner

Secure execution environment for Python code, adapted from Poetiq's arc_agi/sandbox.py.
Runs code in an isolated subprocess with timeout and restricted environment.
Pre-execution AST scanning blocks dangerous imports.
"""

import ast
import asyncio
import os
import sys
import tempfile
import textwrap
from typing import Tuple, Set

# REQ-SEC-01: Blocked modules that pose security risks
BLOCKED_MODULES: Set[str] = {'os', 'sys', 'subprocess', 'shutil', 'socket', 'requests', 'urllib'}


def _scan_for_dangerous_imports(code: str) -> Tuple[bool, str]:
    """
    REQ-SEC-01: Scan code for dangerous imports using AST parsing.
    
    Args:
        code: Python code to scan
        
    Returns:
        Tuple of (is_safe: bool, error_message: str)
        - (True, "") if code is safe
        - (False, "Security Violation: ...") if dangerous imports found
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        # Syntax errors will be caught during execution, allow to pass
        return True, ""
    
    dangerous_found = []
    
    for node in ast.walk(tree):
        # Check 'import X' statements
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name.split('.')[0]  # Get top-level module
                if module_name in BLOCKED_MODULES:
                    dangerous_found.append(f"import {alias.name}")
        
        # Check 'from X import Y' statements
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_name = node.module.split('.')[0]
                if module_name in BLOCKED_MODULES:
                    dangerous_found.append(f"from {node.module} import ...")
        
        # Check __import__ function calls
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == '__import__':
                if node.args and isinstance(node.args[0], ast.Constant):
                    module_name = str(node.args[0].value).split('.')[0]
                    if module_name in BLOCKED_MODULES:
                        dangerous_found.append(f"__import__('{node.args[0].value}')")
    
    if dangerous_found:
        violations = ", ".join(dangerous_found)
        return False, f"Security Violation: Blocked imports detected: {violations}"
    
    return True, ""


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
    # REQ-SEC-01: Pre-execution security scan
    is_safe, security_error = _scan_for_dangerous_imports(code)
    if not is_safe:
        return False, security_error
    
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

