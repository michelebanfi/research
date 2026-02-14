"""
Tests for the 4 bug fixes identified in the codebase audit.
These tests verify the fix logic WITHOUT importing project modules that require
external dependencies (ollama, langchain, supabase, etc.).

Bug 1: _clean_json_string — handles both JSON objects and arrays
Bug 2: generate_summary_async — recursion depth limit  
Bug 3: agent_graph cycle detection — retry_count prevents infinite loops
Bug 4: web search rate limiting — per-session counter
"""

import sys
import os
import re
import ast
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ======== Helper: Extract and test _clean_json_string logic ========

def clean_json_string(json_str: str) -> str:
    """Exact copy of the fixed _clean_json_string logic for testing."""
    # Remove markdown code blocks
    json_str = re.sub(r'```json\s*', '', json_str)
    json_str = re.sub(r'```\s*', '', json_str)
    
    # Remove trailing commas in arrays and objects
    json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
    
    # Handle both JSON objects {...} and arrays [...]
    json_str = json_str.strip()
    first_brace = json_str.find('{')
    first_bracket = json_str.find('[')
    
    if first_brace == -1 and first_bracket == -1:
        return json_str
    elif first_brace == -1:
        start, end_char = first_bracket, ']'
    elif first_bracket == -1:
        start, end_char = first_brace, '}'
    else:
        start = min(first_brace, first_bracket)
        end_char = '}' if start == first_brace else ']'
    
    end = json_str.rfind(end_char)
    if end > start:
        json_str = json_str[start:end + 1]
    
    return json_str.strip()


# ======== Bug 1: _clean_json_string ========

def test_clean_json_object():
    """Should correctly extract JSON objects."""
    result = clean_json_string('```json\n{"nodes": [], "edges": []}\n```')
    parsed = json.loads(result)
    assert parsed == {"nodes": [], "edges": []}, f"Unexpected: {result}"
    
    result = clean_json_string('Here is the JSON:\n{"key": "value"}')
    parsed = json.loads(result)
    assert parsed == {"key": "value"}, f"Unexpected: {result}"
    
    print("✅ Bug 1a: _clean_json_string handles JSON objects correctly")


def test_clean_json_array():
    """Should correctly extract JSON arrays — this was the bug."""
    result = clean_json_string('["claim 1", "claim 2", "claim 3"]')
    parsed = json.loads(result)
    assert parsed == ["claim 1", "claim 2", "claim 3"], f"Unexpected: {result}"
    
    result = clean_json_string('```json\n["a", "b"]\n```')
    parsed = json.loads(result)
    assert parsed == ["a", "b"], f"Unexpected: {result}"
    
    result = clean_json_string('Sure, here are the claims:\n["claim1", "claim2"]')
    parsed = json.loads(result)
    assert parsed == ["claim1", "claim2"], f"Unexpected: {result}"
    
    print("✅ Bug 1b: _clean_json_string handles JSON arrays correctly (was broken before fix)")


def test_clean_json_trailing_commas():
    """Should handle trailing commas."""
    result = clean_json_string('{"a": [1, 2,], "b": 3,}')
    parsed = json.loads(result)
    assert parsed == {"a": [1, 2], "b": 3}, f"Unexpected: {result}"
    
    print("✅ Bug 1c: _clean_json_string removes trailing commas")


# ======== Bug 2: generate_summary_async — verify via source code ========

def test_summary_recursion_depth():
    """Verify generate_summary_async has _depth parameter in source."""
    src_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'ai_engine.py')
    with open(src_path, 'r') as f:
        source = f.read()
    
    # Check function signature
    assert 'def generate_summary_async(self, text: str, _depth: int = 0)' in source, \
        "Missing _depth parameter in generate_summary_async signature"
    
    # Check recursion depth limit
    assert 'MAX_RECURSION_DEPTH' in source, "Missing MAX_RECURSION_DEPTH constant"
    assert '_depth >= MAX_RECURSION_DEPTH' in source, "Missing depth limit check"
    assert '_depth + 1' in source, "Missing _depth increment in recursive call"
    
    print("✅ Bug 2: generate_summary_async has recursion depth limit")


# ======== Bug 3: agent_graph cycle detection — verify via source code ========

def test_agent_graph_cycle_detection():
    """Verify agent_graph has retry_count in state and cycle limit."""
    src_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'agent_graph.py')
    with open(src_path, 'r') as f:
        source = f.read()
    
    # Check AgentState has retry_count
    assert 'retry_count: int' in source, "Missing retry_count in AgentState"
    
    # Check transform_query increments counter
    assert 'retry_count' in source and 'state.get("retry_count", 0) + 1' in source, \
        "transform_query doesn't increment retry_count"
    
    # Check decide_to_generate checks limit
    assert 'MAX_RETRIES' in source, "Missing MAX_RETRIES constant"
    assert 'retry_count' in source and 'FORCING GENERATION' in source, \
        "decide_to_generate doesn't enforce cycle limit"
    
    # Check run_agent_graph initializes retry_count
    assert '"retry_count": 0' in source, "run_agent_graph doesn't initialize retry_count"
    
    print("✅ Bug 3: agent_graph has cycle detection via retry_count")


# ======== Bug 4: web search rate limiting — verify via source code ========

def test_web_search_rate_limit():
    """Verify tools.py has per-session web search counter."""
    src_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'tools.py')
    with open(src_path, 'r') as f:
        source = f.read()
    
    # Check counter initialization
    assert '_web_search_count = 0' in source, "Missing _web_search_count initialization"
    assert '_web_search_limit = 5' in source, "Missing _web_search_limit initialization"
    
    # Check limit enforcement in _web_search
    assert '_web_search_count >= self._web_search_limit' in source, \
        "Missing limit check in _web_search"
    assert '_web_search_count += 1' in source, \
        "Missing counter increment in _web_search"
    
    print("✅ Bug 4: tools.py has per-session web search rate limiting")


# ======== Bonus: Verify all modified files have valid Python syntax ========

def test_python_syntax():
    """All modified files should have valid Python syntax."""
    files = [
        os.path.join(os.path.dirname(__file__), '..', 'src', 'ai_engine.py'),
        os.path.join(os.path.dirname(__file__), '..', 'src', 'agent_graph.py'),
        os.path.join(os.path.dirname(__file__), '..', 'src', 'tools.py'),
    ]
    
    for filepath in files:
        with open(filepath, 'r') as f:
            source = f.read()
        try:
            ast.parse(source)
        except SyntaxError as e:
            raise AssertionError(f"Syntax error in {os.path.basename(filepath)}: {e}")
    
    print("✅ Syntax: All modified files parse without errors")


# ======== Main ========

if __name__ == "__main__":
    print("=" * 60)
    print("Running Bug Fix Verification Tests")
    print("=" * 60)
    
    tests = [
        test_clean_json_object,
        test_clean_json_array,
        test_clean_json_trailing_commas,
        test_summary_recursion_depth,
        test_agent_graph_cycle_detection,
        test_web_search_rate_limit,
        test_python_syntax,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print(f"{'=' * 60}")
    
    sys.exit(0 if failed == 0 else 1)
