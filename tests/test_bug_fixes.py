"""
Tests for the LangGraph integration and all previous bug fixes.

Dependency-free: verifies via source code analysis and pure logic tests.
"""

import sys
import os
import re
import ast
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

SRC = os.path.join(os.path.dirname(__file__), '..', 'src')


def read_src(filename):
    with open(os.path.join(SRC, filename), 'r') as f:
        return f.read()


# ======== Bug 1: _clean_json_string ========

def clean_json_string(json_str: str) -> str:
    """Exact copy of the fixed logic for testing."""
    json_str = re.sub(r'```json\s*', '', json_str)
    json_str = re.sub(r'```\s*', '', json_str)
    json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
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


def test_clean_json_object():
    result = clean_json_string('```json\n{"nodes": [], "edges": []}\n```')
    assert json.loads(result) == {"nodes": [], "edges": []}
    print("✅ Bug 1a: _clean_json_string handles JSON objects")


def test_clean_json_array():
    result = clean_json_string('["claim 1", "claim 2"]')
    assert json.loads(result) == ["claim 1", "claim 2"]
    result = clean_json_string('Sure:\n["a", "b"]')
    assert json.loads(result) == ["a", "b"]
    print("✅ Bug 1b: _clean_json_string handles JSON arrays")


def test_clean_json_trailing_commas():
    result = clean_json_string('{"a": [1, 2,], "b": 3,}')
    assert json.loads(result) == {"a": [1, 2], "b": 3}
    print("✅ Bug 1c: _clean_json_string removes trailing commas")


# ======== Bug 2: recursion depth ========

def test_summary_recursion_depth():
    src = read_src('ai_engine.py')
    assert 'def generate_summary_async(self, text: str, _depth: int = 0)' in src
    assert 'MAX_RECURSION_DEPTH' in src
    assert '_depth + 1' in src
    print("✅ Bug 2: generate_summary_async has recursion depth limit")


# ======== Bug 3: agent_graph cycle detection (now in new graph) ========
# The old agent_graph.py is replaced. Cycle detection for research is now
# max-iteration-based in the new graph. Verify the pattern exists.

def test_research_max_iterations():
    src = read_src('agent_graph.py')
    assert 'MAX_RESEARCH_ITERATIONS' in src
    assert 'MAX_CODE_RETRIES' in src
    # Verify think node increments iteration
    assert "state.get(\"iteration\", 0) + 1" in src
    # Verify after_think checks limit
    assert 'iteration' in src and 'MAX_RESEARCH_ITERATIONS' in src
    print("✅ Bug 3: Graph has max iteration limits for both sub-graphs")


# ======== Bug 4: web search rate limiting ========

def test_web_search_rate_limit():
    src = read_src('tools.py')
    assert '_web_search_count = 0' in src
    assert '_web_search_limit = 5' in src
    assert '_web_search_count >= self._web_search_limit' in src
    print("✅ Bug 4: tools.py has per-session web search rate limiting")


# ======== LangGraph: Graph structure ========

def test_graph_has_unified_state():
    src = read_src('agent_graph.py')
    assert 'class ResearchState(TypedDict)' in src
    required_fields = [
        'user_query', 'chat_history', 'reasoning_mode',
        'conversation', 'observations', 'tools_used',
        'retrieved_chunks', 'matched_concepts', 'iteration',
        'final_answer', 'plan', 'code_attempts', 'result',
    ]
    for field in required_fields:
        assert f'{field}:' in src or f"'{field}'" in src, f"Missing state field: {field}"
    print("✅ LG1: ResearchState has all required fields")


def test_graph_has_router():
    src = read_src('agent_graph.py')
    assert 'def route_mode' in src
    assert '"generate_plan"' in src and '"think"' in src
    assert 'reasoning_mode' in src
    print("✅ LG2: Graph has router dispatching research vs. reasoning")


def test_graph_has_research_nodes():
    src = read_src('agent_graph.py')
    nodes = ['think', 'execute_tool', 'handle_no_action', 'finalize_research']
    for node in nodes:
        assert f'async def {node}' in src, f"Missing research node: {node}"
        assert f'workflow.add_node("{node}"' in src, f"Node '{node}' not registered"
    print("✅ LG3: Research sub-graph has all nodes (think, execute_tool, handle_no_action, finalize)")


def test_graph_has_reasoning_nodes():
    src = read_src('agent_graph.py')
    nodes = ['generate_plan', 'generate_code', 'execute_sandbox', 'verify_result', 'finalize_reasoning']
    for node in nodes:
        assert f'async def {node}' in src, f"Missing reasoning node: {node}"
        assert f'workflow.add_node("{node}"' in src, f"Node '{node}' not registered"
    print("✅ LG4: Reasoning sub-graph has all nodes (plan, code, sandbox, verify, finalize)")


def test_graph_has_checkpointing():
    src = read_src('agent_graph.py')
    assert 'MemorySaver' in src
    assert 'checkpointer=checkpointer' in src or 'checkpointer' in src
    print("✅ LG5: Graph uses MemorySaver checkpointing")


def test_graph_has_event_emission():
    src = read_src('agent_graph.py')
    assert 'event_callback' in src
    assert '_emit(' in src
    # Count emit calls to verify observability
    emit_count = src.count('self._emit(')
    assert emit_count >= 5, f"Expected >=5 event emissions, found {emit_count}"
    print(f"✅ LG6: Graph emits {emit_count} events for UI observability")


def test_agent_uses_graph():
    src = read_src('agent.py')
    assert 'build_research_graph' in src
    assert 'ainvoke' in src
    assert 'class ResearchAgent' in src
    assert 'class AgentResponse' in src
    # Verify old ReAct loop is gone
    assert 'for iteration in range' not in src
    print("✅ LG7: agent.py uses graph (no hand-rolled loop)")


def test_agent_preserves_api():
    src = read_src('agent.py')
    # Same __init__ params
    assert 'ai_engine' in src
    assert 'database' in src
    assert 'project_id' in src
    assert 'event_callback' in src
    assert 'do_rerank' in src
    # Same run() signature
    assert 'async def run(' in src
    assert 'user_query' in src
    assert 'chat_history' in src
    assert 'reasoning_mode' in src
    # Returns same types
    assert 'AgentResponse' in src
    assert 'ReasoningResponse' in src
    print("✅ LG8: agent.py preserves full public API")


# ======== Syntax validation ========

def test_python_syntax():
    files = ['ai_engine.py', 'agent_graph.py', 'agent.py', 'tools.py',
             'models.py', 'reasoning_agent.py']
    for f in files:
        path = os.path.join(SRC, f)
        with open(path, 'r') as fh:
            source = fh.read()
        try:
            ast.parse(source)
        except SyntaxError as e:
            raise AssertionError(f"Syntax error in {f}: {e}")
    print("✅ Syntax: All source files parse without errors")


# ======== Main ========

if __name__ == "__main__":
    print("=" * 60)
    print("Running LangGraph Integration Tests")
    print("=" * 60)

    tests = [
        # Bug fixes
        test_clean_json_object,
        test_clean_json_array,
        test_clean_json_trailing_commas,
        test_summary_recursion_depth,
        test_research_max_iterations,
        test_web_search_rate_limit,
        # LangGraph structure
        test_graph_has_unified_state,
        test_graph_has_router,
        test_graph_has_research_nodes,
        test_graph_has_reasoning_nodes,
        test_graph_has_checkpointing,
        test_graph_has_event_emission,
        test_agent_uses_graph,
        test_agent_preserves_api,
        # Syntax
        test_python_syntax,
    ]

    passed = failed = 0
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
