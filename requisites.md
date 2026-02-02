### **Project Abstract**
> **"Local-Brain, Cloud-Memory Research Assistant"**
>
> A privacy-focused research management interface built with **Streamlit** and **Ollama** that processes documents and Python code using hybrid parsing techniques (Docling + AST). It leverages local LLMs to semantically analyze, summarize, and chunk inputs, while offloading the structured knowledge base and vector storage to **Supabase**. Designed as a high-performance foundation for future evolutionary AI agents, it combines the speed of local inference with the ease of managed cloud databases.

---

### ** Requirements Specification**

**Stack:** Python, Streamlit, Supabase (Postgres+pgvector), Docling, Ollama.

| ID | Category | Requirement Name | Description | Priority |
| :--- | :--- | :--- | :--- | :--- |
| **SYS-01** | System Architecture | **Hybrid Tech Stack** | UI via **Streamlit**, Logic via **Python**, Storage via **Supabase** (Cloud Postgres + pgvector), Inference via **Local Ollama**. | High |
| **SYS-02** | System Architecture | **Async Processing** | Long-running tasks (Ingestion, Parsing, Embedding) must run asynchronously to prevent UI freezing while waiting for Supabase API calls. | High |
| **SYS-03** | System Architecture | **Hybrid AI Pipeline** | Use **Ollama (Local)** for free/private embedding and summarization generation, then push results to **Supabase (Cloud)** for storage. | High |
| **PM-01** | Project Management | **Project Isolation** | Data queries to Supabase must always be filtered by `project_id` to ensure strict separation between different research contexts. | High |
| **PM-02** | Project Management | **Cascading Deletion** | Deleting a Project must trigger a cascading delete in Supabase to remove all linked Files, Vectors, and Metadata. | High |
| **ING-01** | Ingestion Pipeline | **File Router** | System must automatically route files: **Docling** for Docs (PDF/MD/DOCX) and **AST Parser** for Code (.py). | High |
| **ING-02** | Ingestion Pipeline | **Semantic Code Parsing** | Python files must be parsed using AST to split chunks by logical boundaries (Classes/Functions) before uploading to Supabase. | High |
| **ING-03** | Ingestion Pipeline | **Docling Integration** | Non-code documents must be parsed via **Docling**, converted to Markdown, and chunked respecting the layout hierarchy. | High |
| **AI-01** | AI Processing | **Auto-Summarization** | Local LLM generates a technical summary for every file; this text is stored in the `Files` table in Supabase. | High |
| **AI-02** | AI Processing | **Keyword Extraction** | Local LLM extracts keywords; these are normalized and stored in a `Keywords` table to link related files. | High |
| **AI-03** | AI Processing | **Vector Embedding** | Text chunks are embedded locally (e.g., Qwen/Nomic) and pushed to the Supabase `vector` column. | High |
| **DB-01** | Database | **Supabase Client** | Use the `supabase-py` client to manage connection, insertion, and vector similarity search (RPC calls). | High |
| **DB-02** | Database | **Many-to-Many Linking** | Schema must relate `Files` <-> `Keywords` to allow graph-style traversal (finding files with shared concepts). | Medium |
| **DB-03** | Database | **Atomic Updates** | When updating a file, the system must transactionally delete old rows in Supabase before inserting new ones to prevent duplication. | High |
| **UI-01** | User Interface | **Status Monitor** | UI must display real-time status (e.g., "Uploading to Cloud...", "Indexing Vectors"). | Medium |
| **UI-02** | User Interface | **Knowledge Explorer** | A generic view to inspect the Supabase tables (Files/Chunks) directly from the Streamlit dashboard. | Medium |
| **FUT-01** | Future Proofing | **Version Columns** | Include `version_id` or `created_at` in Supabase tables to support future evolutionary algorithms. | Low |