# Project Roadmap & Development History

This document outlines the original vision, implementation status, and future direction of the **Local-Brain Research Assistant**.

---

## Phase 1: Foundation (The "RAG" Phase)
**Original Idea:** 
Build a local knowledge base that allows users to chat with their documents (PDFs, Code) using an LLM.

### Implemented Features
- **Ingestion Pipeline**: Built `DoclingParser` to extract text from PDFs.
- **Vector Database**: Integrated Supabase (`pgvector`) to store semantic chunks.
- **Basic RAG**: Implemented simple cosine similarity search (`vector_search`).
- **UI**: Created a Streamlit interface for file upload and chat.

---

## Phase 2: Agent Intelligence (The "Reasoning" Phase)
**Original Idea:** 
Transform the system from a passive "search engine" into an active "agent" that can reason, plan, and use tools.

### Implemented Features
- **ReAct Loop**: Built `ResearchAgent` in `src/agent.py` to loop through Thought -> Action -> Observation.
- **Tool Registry**: Created `ToolRegistry` to manage tools like `vector_search` and `graph_search`.
- **Reasoning Mode**: Added a "Plan & Code" mode (`src/reasoning_agent.py`) that can write and execute Python code in a sandbox for data analysis.
- **GraphRAG**: Added initial support for Knowledge Graph queries (Nodes & Edges).

---

## Phase 3: Performance, Scale & Observability (The "Optimization" Phase)
**Original Idea:** 
Make the system faster, smarter at retrieval, and transparent to the user.

### Implemented Features
- **Hierarchical Ingestion**: Updated `src/ingestion.py` to preserve document structure (Parent Sections -> Child Text) for "Small-to-Big" retrieval.
- **Advanced Retrieval**: Implemented `retrieve_advanced` in `AIEngine` with:
    - **Query Expansion**: Generating synonyms/hyponyms.
    - **RRF (Reciprocal Rank Fusion)**: Merging results from multiple queries.
    - **Re-Ranking**: Using FlashRank to sort top candidates.
- **Caching**: Added persistent disk caching (`src/cache.py`) for embeddings to speed up queries.
- **UI Observability**: Built a Split-Screen UI (`src/ui/chat.py`, `src/ui/process.py`) to visualize the agent's live thought process, tool inputs, and search results.

---

## Phase 4: LangGraph Integration (The "Workflow" Phase)
**Development Direction:** 
Move from a simple loop to a structured state machine for complex, multi-step workflows.

### Planned Features
- **State Management**: Replace custom `ResearchAgent` loop with LangGraph for robust state handling.
- **Checkpoints**: Ability to pause/resume research tasks.
- **Human-in-the-loop**: Allow users to approve/edit the agent's plan before execution.
- **Sub-Graphs**: Dedicated sub-graphs for different tasks (e.g., "Research Sub-graph", "Coding Sub-graph").

---

## Phase 5: Multi-Agent Collaboration (The "Team" Phase)
**Development Direction:** 
Scale to a team of specialized agents working together.

### Sketch
- **Manager Agent**: Breaks down high-level user goals into sub-tasks.
- **Researcher Agent**: Scours the knowledge base and web for information.
- **Coder Agent**: Writes and tests code (building on Phase 2).
- **Critic Agent**: Reviews outputs for accuracy and hallucinations.
- **Shared Memory**: A shared "Blackboard" or database for agents to exchange findings.
