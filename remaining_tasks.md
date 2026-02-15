# Remaining Implementations

Based on the original plan, current codebase state, and recent LLM analyses (DeepSeek, Grok, Kimi, Zai), the following features and improvements are pending.

## 1. Performance & Infrastructure (Critical)
- [ ] **Fix Asyncio Event Loop Issues**:
    - Current ad-hoc usage of `nest_asyncio` and `asyncio.run` in Streamlit causes instability.
    - **Goal**: Standardize on a robust async pattern (e.g., `get_or_create_event_loop`) or migrate to Streamlit's native async support.
    - *Reference*: `src/app.py`, `src/ai_engine.py`

- [ ] **Optimize Re-ranking & Caching**:
    - The `FlashRank` model is re-instantiated on every call, and embeddings are cached with simple MD5.
    - **Goal**: Implement Singleton pattern for `Ranker`.
    - **Goal**: Implement LRU Cache with semantic hashing for embeddings to improve hit rates.
    - *Reference*: `src/ai_engine.py`

- [ ] **Database Query Optimization**:
    - Search currently involves multiple round-trips (vector match -> metadata fetch).
    - **Goal**: Consolidate into a single RPC call or optimized query to reduce latency.
    - *Reference*: `src/database.py`

## 2. Agent Reliability & Reasoning (The "Brain")
- [ ] **Dynamic Query Decomposition**:
    - **Goal**: Instead of just synonym expansion, break complex user queries into sub-questions (e.g., "Compare X and Y" -> "What is X?", "What is Y?", "Differences?").
    - *Reference*: `src/ai_engine.py`

- [ ] **Explicit Critique/Self-Correction Node**: 
    - **Goal**: Implement a `critique` node in `agent_graph.py` to evaluate retrieval quality *before* generating an answer. If retrieval is poor, trigger `web_search` or query reformulation.
    - *Reference*: `src/agent_graph.py`

- [ ] **Context Compression & Management**:
    - "Lost in the Middle" problem: naive context stuffing hurts reasoning.
    - **Goal**: Implement "Context Distillation" where chunks are summarized/compressed before being passed to the final prompt.
    - *Reference*: `src/agent_graph.py`, `src/ai_engine.py`

## 3. Knowledge Organization (Graph & Data)
- [ ] **Hierarchical Knowledge Graph**:
    - Current graph is flat.
    - **Goal**: Implement a hierarchy: `Project` -> `File` -> `Section` -> `Chunk` -> `Concept`.
    - **Goal**: Use this hierarchy for "Zoom In/Out" retrieval (start with summary, drill down to details).
    - *Reference*: `schema.sql`, `src/ingestion.py`

- [ ] **Semantic Chunking with Overlap**:
    - Current chunking might break semantic context.
    - **Goal**: Enhance `_split_text_semantically` to better respect document structure (headers, paragraphs) and include overlap/context windows.
    - *Reference*: `src/ai_engine.py`

- [ ] **Type-Strict Graph Extraction & Normalization**:
    - **Goal**: Enforce strict node types (Concept, Tool, Metric, etc.) and improve entity normalization (merging "LLM" and "Large Language Model") using the LLM logic to prevent graph pollution.
    - *Reference*: `src/ai_engine.py`

## 4. Advanced Features (UI & UX)
- [ ] **Streaming Responses**:
    - **Goal**: Stream agent thoughts and final answers to the UI for better perceived latency.
    - *Reference*: `src/ui/chat.py`, `src/agent.py`

- [ ] **UI for Semantic Clustering**:
    - **Goal**: Add a "Knowledge Clusters" view in the Graph tab to visualize topic clusters.
    - *Reference*: `src/ui/graph_view.py`

## 5. Testing & Verification
- [ ] **Evaluation Pipeline**:
    - **Goal**: Create a standard evaluation set (qa_pairs.json) and a script (`evaluation.py`) to measure RAG performance (Precision/Recall, Answer Relevancy).
    - *Reference*: `tests/evaluation.py`

## 6. Security & Stability
- [ ] **Sandbox Security Hardening**:
    - Current AST-based blocking in `sandbox.py` might be bypassable.
    - **Goal**: Enhance code sanitization or move to a more isolated environment (e.g., Docker, gVisor, or restrictedpython) if possible.
    - *Reference*: `src/sandbox.py`

- [ ] **Ingestion Pipeline Optimization**:
    - Large file ingestion blocks the UI.
    - **Goal**: Move ingestion to a background task queue (or separate thread/process) and provide progress updates to the UI.
    - *Reference*: `src/ui/ingest.py`

