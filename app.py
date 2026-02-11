
import os
import streamlit as st
import asyncio
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Fix event loop issues in Streamlit
import nest_asyncio
nest_asyncio.apply()

from streamlit_agraph import agraph, Node, Edge, Config as AgraphConfig
from src.database import DatabaseClient
from src.ingestion import FileRouter, DoclingParser, ASTParser, FileType
from src.ai_engine import AIEngine

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Local-Brain Research Assistant", layout="wide")

# --- Cached Resources ---
@st.cache_resource
def get_db_client():
    try:
        return DatabaseClient()
    except Exception as e:
        st.error(f"Failed to connect to Database: {e}. Please check .env and Supabase.")
        return None

@st.cache_resource
def get_ai_engine():
    return AIEngine()

db = get_db_client()
ai = get_ai_engine()

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_context" not in st.session_state:
    st.session_state.last_context = []
if "selected_project_id" not in st.session_state:
    st.session_state.selected_project_id = None
if "graph_nodes" not in st.session_state:
    st.session_state.graph_nodes = []
if "graph_edges" not in st.session_state:
    st.session_state.graph_edges = []
if "matched_concepts" not in st.session_state:
    st.session_state.matched_concepts = []  # REQ-01: GraphRAG matched concepts

# --- Main Layout ---
st.title("Local-Brain Research Assistant")

# --- Collapsible Project Sidebar ---
with st.sidebar:
    st.header("📁 Projects")
    
    if db:
        projects = db.get_projects()
        
        if projects:
            project_names = [p['name'] for p in projects]
            project_map = {p['name']: p['id'] for p in projects}
            
            # Find current index
            current_idx = 0
            if st.session_state.get("selected_project_id"):
                for i, p in enumerate(projects):
                    if p['id'] == st.session_state.selected_project_id:
                        current_idx = i
                        break
            
            selected_name = st.selectbox(
                "Select Project",
                project_names,
                index=current_idx,
                key="project_selector"
            )
            
            # Update session state if project changed
            new_project_id = project_map[selected_name]
            if new_project_id != st.session_state.selected_project_id:
                st.session_state.selected_project_id = new_project_id
                # Clear chat history when project changes
                st.session_state.chat_history = []
                st.session_state.last_context = []
                st.session_state.graph_nodes = []
                st.session_state.graph_edges = []
                st.rerun()
            
            st.caption(f"ID: `{st.session_state.selected_project_id}`")
        else:
            st.info("No projects yet.")
            st.session_state.selected_project_id = None
        
        # Create new project
        with st.expander("➕ Create New Project"):
            new_project_name = st.text_input("Project Name", key="new_project_input")
            if st.button("Create", key="create_project_btn"):
                if new_project_name:
                    res = db.create_project(new_project_name)
                    if res:
                        st.success(f"Created '{new_project_name}'!")
                        st.rerun()
                    else:
                        st.error("Failed to create project.")
    else:
        st.error("Database not connected.")

# --- Main Content Area with Tabs ---
if st.session_state.get("selected_project_id"):
    tab_chat, tab_ingest, tab_graph = st.tabs(["💬 Chat", "📥 Ingest Files", "🕸️ Knowledge Graph"])
    
    # ==================== CHAT TAB ====================
    with tab_chat:
        st.subheader("Chat with your Knowledge Base")
        
        # Two-column layout: Chat (70%) | Context Panel (30%)
        chat_col, context_col = st.columns([7, 3])
        
        with chat_col:
            # Chat settings
            with st.expander("⚙️ Chat Settings", expanded=False):
                do_rerank = st.checkbox("Enable Re-ranking", value=True, help="Uses FlashRank to improve context relevance")
                top_k = st.slider("Number of context chunks", min_value=3, max_value=10, value=5)
                reasoning_mode = st.toggle(
                    "🧠 Reasoning Mode", 
                    value=False, 
                    help="Enable Plan & Code mode: the model will plan, write code, execute it in a sandbox, and iterate on errors"
                )
            
            # Display chat history
            chat_container = st.container()
            with chat_container:
                for msg in st.session_state.chat_history:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask a question about your knowledge base..."):
                # Add user message to history
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    # REQ-UI-02: Status container for tool execution visibility
                    status_container = st.status("🤔 Thinking...", expanded=True)
                    
                    # Track tool usage for status updates
                    tools_used = []
                    
                    def update_status(tool_name: str, phase: str):
                        """Callback for tool execution visibility."""
                        if phase == "start":
                            status_container.update(label=f"🔧 Using {tool_name}...")
                            status_container.write(f"📍 Calling `{tool_name}`...")
                        elif phase == "complete":
                            status_container.write(f"✅ `{tool_name}` complete")
                            tools_used.append(tool_name)
                    
                    try:
                        # REQ-UI-01: Single agent.run() call replaces determine_intent + if/else
                        from src.agent import ResearchAgent
                        
                        agent = ResearchAgent(
                            ai_engine=ai,
                            database=db,
                            project_id=st.session_state.selected_project_id,
                            status_callback=update_status,
                            do_rerank=do_rerank
                        )
                        
                        # Run the agent
                        result = asyncio.run(agent.run(
                            prompt,
                            st.session_state.chat_history[:-1],  # Exclude current message
                            reasoning_mode=reasoning_mode
                        ))
                        
                        # Update status to complete
                        if tools_used:
                            status_container.update(label=f"✨ Done (used: {', '.join(tools_used)})", state="complete", expanded=False)
                        else:
                            status_container.update(label="✨ Done", state="complete", expanded=False)
                        
                        # Check if it's a ReasoningResponse or AgentResponse
                        from src.models import ReasoningResponse
                        
                        if isinstance(result, ReasoningResponse):
                            # Reasoning mode response
                            if result.plan:
                                with st.expander("📋 Plan", expanded=False):
                                    st.markdown(f"**Goal:** {result.plan.goal}")
                                    st.markdown(f"**Context needed:** {result.plan.context_needed}")
                                    st.markdown(f"**Verification:** {result.plan.verification_logic}")
                            
                            # Show attempts
                            if result.attempts:
                                with st.expander(f"🔄 Code Attempts ({len(result.attempts)})", expanded=False):
                                    for attempt in result.attempts:
                                        status_icon = "✅" if attempt.success else "❌"
                                        st.markdown(f"**Attempt {attempt.attempt_number}** {status_icon}")
                                        st.code(attempt.code, language="python")
                                        if attempt.output:
                                            st.text(f"Output: {attempt.output[:500]}")
                            
                            # Final answer
                            if result.success:
                                assistant_message = f"**Result:**\n```\n{result.final_output}\n```"
                            else:
                                assistant_message = f"❌ {result.error}\n\nLast output:\n```\n{result.final_output or 'No output'}\n```"
                        else:
                            # Standard AgentResponse
                            # Store context for display panel
                            st.session_state.last_context = result.retrieved_chunks
                            st.session_state.matched_concepts = result.matched_concepts
                            assistant_message = result.answer
                        
                    except Exception as e:
                        status_container.update(label="❌ Error", state="error")
                        assistant_message = f"I encountered an error: {str(e)}"
                        import traceback
                        print(f"Agent error: {traceback.format_exc()}")
                    
                    st.markdown(assistant_message)
                    st.session_state.chat_history.append({"role": "assistant", "content": assistant_message})
            
            # Clear chat button
            if st.session_state.chat_history:
                if st.button("🗑️ Clear Chat", key="clear_chat"):
                    st.session_state.chat_history = []
                    st.session_state.last_context = []
                    st.rerun()
        
        with context_col:
            st.markdown("### 📚 Retrieved Context")
            
            # REQ-01: Show matched graph concepts
            if st.session_state.get('matched_concepts'):
                with st.expander("🕸️ Graph-Matched Concepts", expanded=True):
                    concepts = st.session_state.matched_concepts[:8]  # Limit display
                    st.markdown(" • ".join([f"`{c}`" for c in concepts]))
            
            if st.session_state.last_context:
                for i, chunk in enumerate(st.session_state.last_context):
                    # Determine source type
                    source_type = chunk.get('source', 'vector')
                    source_icon = "🔍" if source_type == 'vector' else "🕸️"
                    
                    with st.expander(f"{source_icon} Source {i+1}", expanded=(i == 0)):
                        # Show scores and source
                        original_sim = chunk.get('similarity', 0)
                        rerank_score = chunk.get('rerank_score')
                        
                        # Source badge
                        source_badge = "Vector Search" if source_type == 'vector' else "Graph Retrieval"
                        st.caption(f"**{source_badge}**")
                        
                        if rerank_score is not None:
                            st.caption(f"Original: `{original_sim:.3f}` → Re-ranked: `{rerank_score:.3f}`")
                        else:
                            st.caption(f"Similarity: `{original_sim:.3f}`")
                        
                        st.markdown(chunk.get('content', '')[:500] + ("..." if len(chunk.get('content', '')) > 500 else ""))
                        
                        # REQ-VIS-04 / REQ-VIS-05: View Source Buttons
                        file_path = chunk.get('file_path')
                        metadata = chunk.get('metadata', {})
                        
                        if file_path:
                            # Construct local static link (relative to static folder)
                            # Assuming app serves from root, and file_path is absolute or relative
                            # We stored 'static/uploads/filename' in database during ingestion below
                            
                            # Clean up path to be relative to 'static/' if needed
                            if "static/" in file_path:
                                relative_path = file_path.split("static/")[1]
                            else:
                                # Fallback: Check if file exists in static/uploads/
                                possible_upload_path = Path("static/uploads") / file_path
                                if possible_upload_path.exists():
                                    relative_path = f"uploads/{file_path}"
                                else:
                                    relative_path = os.path.basename(file_path) 
                                
                            file_url = f"app/static/{relative_path}"
                            
                            # Layout buttons
                            col1, col2 = st.columns([1, 1])
                            
                            # 1. View PDF
                            if file_path.lower().endswith('.pdf'):
                                page = metadata.get('page_number', 1)
                                with col1:
                                    st.link_button("📄 View PDF", f"{file_url}#page={page}")
                                    
                            # 2. View Code / Text
                            elif file_path.lower().endswith(('.py', '.md', '.txt')):
                                start_line = metadata.get('start_line')
                                end_line = metadata.get('end_line')
                                
                                with col2:
                                    if st.button("📝 View Code", key=f"view_{i}_{chunk.get('id')}"):
                                        with st.dialog("Source Code", width="large"):
                                            try:
                                                with open(file_path, 'r') as f:
                                                    lines = f.readlines()
                                                    
                                                # Highlight specific lines
                                                if start_line and end_line:
                                                    # Provide some context around lines
                                                    ctx_start = max(0, start_line - 5)
                                                    ctx_end = min(len(lines), end_line + 5)
                                                    
                                                    code_segment = "".join(lines[ctx_start:ctx_end])
                                                    st.caption(f"Showing lines {ctx_start+1} - {ctx_end}")
                                                    # Use emphasize_lines syntax if supported or just show block
                                                    st.code(code_segment, language='python' if file_path.endswith('.py') else 'markdown')
                                                else:
                                                    st.code("".join(lines), language='python' if file_path.endswith('.py') else 'markdown')
                                                    
                                            except Exception as e:
                                                st.error(f"Could not read file: {e}")
            else:
                st.info("Ask a question to see which knowledge chunks are used.")
    
    # ==================== INGEST TAB ====================
    with tab_ingest:
        st.subheader("Ingest Documents & Code")
        
        # File browser
        files_col, upload_col = st.columns([1, 1])
        
        with files_col:
            st.markdown("### 📂 Current Files")
            files = db.get_project_files(st.session_state.selected_project_id)
            
            if files:
                for f in files:
                    with st.expander(f"📄 {f['name']}"):
                        st.write(f"**Summary:** {f['summary'][:200]}..." if f.get('summary') and len(f['summary']) > 200 else f.get('summary', 'No summary'))
                        st.write(f"**Keywords:** {', '.join(f['metadata'].get('keywords', [])[:5])}")
                        st.caption(f"Processed: {f['processed_at']}")
                        
                        # Related files
                        related = db.get_related_files(f['id'])
                        if related:
                            st.write("**Related:**")
                            for rf in related[:3]:
                                st.caption(f"• {rf['name']} ({rf['shared_count']} shared)")
                        
                        if st.button("🗑️ Delete", key=f"del_{f['id']}"):
                            if db.delete_file(f['id']):
                                st.success("Deleted!")
                                st.rerun()
                            else:
                                st.error("Delete failed.")
            else:
                st.info("No files ingested yet.")
        
        with upload_col:
            st.markdown("### 📤 Upload New File")
            
            uploaded_file = st.file_uploader(
                "Upload Document or Code", 
                type=['pdf', 'md', 'txt', 'py', 'docx'],
                key="file_uploader"
            )
            
            if uploaded_file and st.button("🚀 Process & Ingest", key="ingest_btn"):
                with st.spinner("Processing..."):
                    # Save to temp file
                    # REQ-VIS-02: Save to static/uploads
                    upload_dir = Path("static/uploads")
                    upload_dir.mkdir(parents=True, exist_ok=True)
                    
                    file_path = upload_dir / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                        
                    tmp_path = str(file_path)
                    
                    try:
                        # 1. Route
                        ftype = FileRouter.route(tmp_path)
                        st.info(f"Detected file type: {ftype.value}")
                        
                        # 2. Parse
                        chunks = []
                        code_graph_data = {"nodes": [], "edges": []}  # REQ-INGEST-01: For code files
                        
                        if ftype == FileType.CODE:
                            parser = ASTParser()
                            # REQ-INGEST-01: ASTParser now returns dict with chunks and graph_data
                            parse_result = parser.parse(tmp_path)
                            chunks = parse_result.get("chunks", [])
                            code_graph_data = parse_result.get("graph_data", {"nodes": [], "edges": []})
                        elif ftype == FileType.DOCUMENT:
                            parser = DoclingParser()
                            chunks = parser.parse(tmp_path)
                        else:
                            st.warning("Unsupported file type.")
                        
                        st.write(f"Parsed {len(chunks)} chunks.")
                        
                        # REQ-INGEST-01: Show code graph preview for Python files
                        if ftype == FileType.CODE and code_graph_data["nodes"]:
                            with st.expander("🔍 Code Structure Preview", expanded=True):
                                classes = [n for n in code_graph_data["nodes"] if n["type"] == "Class"]
                                methods = [n for n in code_graph_data["nodes"] if n["type"] == "Method"]
                                functions = [n for n in code_graph_data["nodes"] if n["type"] == "Function"]
                                
                                st.markdown(f"**Classes:** {len(classes)} | **Methods:** {len(methods)} | **Functions:** {len(functions)}")
                                
                                # Show class hierarchy
                                if classes:
                                    st.markdown("📦 **Classes:**")
                                    for c in classes[:10]:
                                        # Find methods for this class
                                        class_methods = [e for e in code_graph_data.get("edges", []) 
                                                        if e.get("source") == c["name"] and e.get("relation") == "contains"]
                                        method_names = [e["target"] for e in class_methods[:5]]
                                        if method_names:
                                            st.markdown(f"  - `{c['name']}` → {', '.join([f'`{m}`' for m in method_names])}")
                                        else:
                                            st.markdown(f"  - `{c['name']}`")
                                
                                # Show inheritance
                                inherits = [e for e in code_graph_data.get("edges", []) if e.get("relation") == "inherits"]
                                if inherits:
                                    st.markdown("🔗 **Inheritance:**")
                                    for e in inherits[:5]:
                                        st.markdown(f"  - `{e['source']}` extends `{e['target']}`")
                        
                        # UI: Document parsing preview for PDFs/Docs
                        elif ftype == FileType.DOCUMENT and chunks:
                            with st.expander("📄 Document Preview", expanded=True):
                                # Show section structure
                                sections = set()
                                for chunk in chunks:
                                    headings = chunk.get("metadata", {}).get("headings", [])
                                    for h in headings:
                                        if h:
                                            sections.add(h)
                                
                                if sections:
                                    st.markdown("**Detected Sections:**")
                                    st.markdown(" | ".join([f"`{s}`" for s in list(sections)[:10]]))
                                
                                # Show first chunk as sample
                                st.markdown("---")
                                st.markdown("**Sample Content (first chunk):**")
                                sample = chunks[0].get("content", "")[:800]
                                if len(chunks[0].get("content", "")) > 800:
                                    sample += "..."
                                st.markdown(sample)
                        
                        # 3. AI Processing (Async & Parallel)
                        st.text("Running AI Analysis (Summary, Graph, Embeddings)...")
                        
                        full_text = "\n".join([c['content'] for c in chunks])
                        
                        # Fix for event loop issues with Streamlit
                        import nest_asyncio
                        try:
                            nest_asyncio.apply()
                        except RuntimeError:
                            pass  # Already applied
                        
                        async def run_ingestion_pipeline():
                            # Create fresh async client to avoid closed loop issues
                            import ollama
                            async_client = ollama.AsyncClient()
                            
                            async def safe_summary():
                                try:
                                    # Use AIEngine's summarize method which handles the remote model correctly
                                    return await ai.generate_summary_async(full_text[:4000])
                                except Exception as e:
                                    print(f"Summary error: {e}")
                                    return "Summary generation failed."
                            
                            async def safe_graph():
                                try:
                                    return await ai.extract_metadata_graph_async(full_text)
                                except Exception as e:
                                    print(f"Graph error: {e}")
                                    return {"nodes": [], "edges": []}
                            
                            async def safe_embedding(text):
                                try:
                                    response = await async_client.embeddings(model=ai.embed_model, prompt=text)
                                    return response["embedding"]
                                except Exception as e:
                                    print(f"Embedding error: {e}")
                                    return []
                            
                            # Run all tasks
                            summary_result = await safe_summary()
                            graph_result = await safe_graph()
                            embedding_results = await asyncio.gather(
                                *[safe_embedding(c['content']) for c in chunks]
                            )
                            
                            return summary_result, graph_result, embedding_results
                        
                        # Use existing event loop if available, else create new
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_closed():
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                            summary, graph_data, embeddings = loop.run_until_complete(run_ingestion_pipeline())
                        except RuntimeError:
                            # Fallback for when there's no event loop
                            summary, graph_data, embeddings = asyncio.run(run_ingestion_pipeline())
                        
                        # Validate embeddings - filter out failed ones
                        valid_chunks = []
                        for i, emb in enumerate(embeddings):
                            if emb and len(emb) > 0:
                                chunks[i]['embedding'] = emb
                                valid_chunks.append(chunks[i])
                            else:
                                st.warning(f"Chunk {i} had empty embedding, skipping.")
                        
                        chunks = valid_chunks
                        st.write(f"Generated {len(chunks)} valid embeddings.")
                        st.text_area("Summary", summary, height=100)
                        
                        # Graph Data
                        # REQ-INGEST-01: Merge code-extracted graph with AI-extracted graph
                        nodes = graph_data.get('nodes', [])
                        edges = graph_data.get('edges', [])
                        
                        # Add code structure nodes/edges (for Python files)
                        if code_graph_data["nodes"]:
                            nodes.extend(code_graph_data["nodes"])
                            edges.extend(code_graph_data["edges"])
                        
                        st.write(f"Extracted {len(nodes)} nodes and {len(edges)} edges.")
                        
                        keywords = [n['name'] for n in nodes]
                        
                        # 4. Storage
                        st.text("Storing data...")
                        
                        file_meta = db.upload_file_metadata(
                            project_id=st.session_state.selected_project_id,
                            name=uploaded_file.name,
                            path=str(file_path),
                            summary=summary,
                            metadata={"keywords": keywords}
                        )
                        
                        if file_meta:
                            file_id = file_meta['id']
                            db.store_chunks(file_id, chunks)
                            db.store_keywords(file_id, keywords)
                            db.store_graph_data(file_id, nodes, edges)
                            st.success("✅ Ingestion Complete!")
                        else:
                            st.error("Failed to upload file metadata.")
                            
                    except Exception as e:
                        st.error(f"Error during ingestion: {e}")
                        import traceback
                        st.text(traceback.format_exc())
                    # allow file to persist in static/uploads

    
    # ==================== KNOWLEDGE GRAPH TAB ====================
    with tab_graph:
        st.subheader("Knowledge Graph Explorer")
        
        st.caption("Visualizing the global knowledge graph for your project.")
        
        if st.button("🔄 Load/Refresh Graph", key="load_graph_btn"):
            with st.spinner("Loading graph data..."):
                graph_data_rows = db.get_project_graph(st.session_state.selected_project_id, limit=500)
                
                nodes = []
                edges = []
                added_nodes = set()
                
                for row in graph_data_rows:
                    s_name = row['source_name']
                    t_name = row['target_name']
                    s_type = row['source_type']
                    t_type = row['target_type']
                    relation = row['edge_type']
                    
                    # Color by type
                    type_colors = {
                        "Concept": "#FF6B6B",
                        "Tool": "#4ECDC4",
                        "System": "#45B7D1",
                        "Metric": "#96CEB4",
                        "Person": "#DDA0DD",
                    }
                    
                    if s_name not in added_nodes:
                        color = type_colors.get(s_type, "#95A5A6")
                        nodes.append(Node(id=s_name, label=s_name, size=20, color=color))
                        added_nodes.add(s_name)
                    
                    if t_name not in added_nodes:
                        color = type_colors.get(t_type, "#95A5A6")
                        nodes.append(Node(id=t_name, label=t_name, size=20, color=color))
                        added_nodes.add(t_name)
                    
                    edges.append(Edge(source=s_name, target=t_name, label=relation))
                
                st.session_state.graph_nodes = nodes
                st.session_state.graph_edges = edges
        
        if st.session_state.graph_nodes:
            st.success(f"Showing {len(st.session_state.graph_nodes)} nodes and {len(st.session_state.graph_edges)} edges.")
            
            # Legend
            st.markdown("""
            **Legend:** 
            🔴 Concept | 
            🔵 Tool | 
            💙 System | 
            💚 Metric | 
            💜 Person
            """)
            
            config = AgraphConfig(
                width=900, 
                height=600, 
                directed=True, 
                nodeHighlightBehavior=True, 
                highlightColor="#F7A7A6",
                collapsible=True,
                physics=True
            )
            
            clicked_node = agraph(
                nodes=st.session_state.graph_nodes, 
                edges=st.session_state.graph_edges, 
                config=config
            )
            
            if clicked_node:
                st.info(f"Selected: **{clicked_node}**")
        else:
            st.info("Click 'Load/Refresh Graph' to visualize your knowledge base.")

else:
    st.info("👈 Please select or create a project from the sidebar to get started.")
