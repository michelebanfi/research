
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
            if st.session_state.selected_project_id:
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
if st.session_state.selected_project_id:
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

                    with st.spinner("Thinking..."):

                        intent = asyncio.run(ai.determine_intent(prompt))
                    
                    print(intent)

                    if intent == "BROAD":
                        with st.spinner("Retrieving project overview..."):
                            st.caption("🔍 Detected intent: **Broad Overview** (Using File Summaries)")
                            # TOOL USE: Fetch pre-computed summaries instead of vector search
                            summaries_text = db.get_all_file_summaries(st.session_state.selected_project_id)
                            
                            # Package it to look like a chunk so existing chat function accepts it
                            context_chunks = [{
                                "content": summaries_text,
                                "source": "database_summary",
                                "similarity": 1.0
                            }]
                            
                    else:
                        with st.spinner("Searching knowledge base (Vector + GraphRAG)..."):
                            # 1. Generate query embedding
                            query_embedding = ai.generate_embedding(prompt)
                        
                        if query_embedding:
                            # 2. Vector search - get initial results
                            vector_results = db.search_vectors(
                                query_embedding, 
                                match_threshold=0.3, 
                                project_id=st.session_state.selected_project_id,
                                match_count=top_k * 2  # Get more for combining with graph
                            )
                            
                            # Mark vector results with source
                            for r in vector_results:
                                r['source'] = 'vector'
                            
                            # =============================================
                            # REQ-01: GraphRAG - Augment with graph context
                            # =============================================
                            graph_results = []
                            matched_concepts = []
                            
                            try:
                                # Extract key terms from query (simple tokenization)
                                import re
                                query_terms = [w.lower() for w in re.findall(r'\b\w{3,}\b', prompt)]
                                
                                # Search for matching concepts in the knowledge graph
                                if query_terms:
                                    matching_nodes = db.search_nodes_by_name(
                                        query_terms, 
                                        st.session_state.selected_project_id,
                                        limit=10
                                    )
                                    
                                    if matching_nodes:
                                        matched_concepts = [n['name'] for n in matching_nodes]
                                        node_ids = [n['id'] for n in matching_nodes]
                                        
                                        # Also get related concepts (1-hop neighbors)
                                        related_concepts = db.get_related_concepts(node_ids, max_hops=1)
                                        if related_concepts:
                                            node_ids.extend([n['id'] for n in related_concepts])
                                            matched_concepts.extend([n['name'] for n in related_concepts[:5]])
                                        
                                        # Get chunks connected to these concepts
                                        graph_results = db.get_chunks_by_concepts(
                                            node_ids,
                                            st.session_state.selected_project_id,
                                            limit=top_k
                                        )
                            except Exception as e:
                                print(f"GraphRAG error (non-fatal): {e}")
                            
                            # Combine results: vector + graph (deduplicated)
                            seen_chunk_ids = set()
                            combined_results = []
                            
                            # Add vector results first
                            for r in vector_results:
                                chunk_id = r.get('id')
                                if chunk_id and chunk_id not in seen_chunk_ids:
                                    seen_chunk_ids.add(chunk_id)
                                    combined_results.append(r)
                            
                            # Add graph results that weren't in vector results
                            for r in graph_results:
                                chunk_id = r.get('id')
                                if chunk_id and chunk_id not in seen_chunk_ids:
                                    seen_chunk_ids.add(chunk_id)
                                    # Graph results don't have similarity scores, assign a default
                                    r['similarity'] = 0.5  # Neutral score for re-ranking
                                    combined_results.append(r)
                            
                            # 3. Re-rank combined results if enabled
                            if do_rerank and combined_results:
                                results = ai.rerank_results(prompt, combined_results, top_k=top_k)
                            else:
                                results = combined_results[:top_k]
                            
                            # Store context for display (including graph info)
                            st.session_state.last_context = results
                            st.session_state.matched_concepts = matched_concepts  # For display

                            context_chunks = results
                        
                        else:
                            assistant_message = "Failed to process your question. Please try again."
                            st.session_state.last_context = []
                            st.session_state.matched_concepts = []
                            
                        # 4. Generate response with RAG
                    if context_chunks:
                        response = ai.chat_with_context(
                            prompt, 
                            context_chunks, 
                            st.session_state.chat_history[:-1]  # Exclude current message
                        )
                        assistant_message = response.get("response", "I couldn't generate a response.")
                    else:
                        assistant_message = "I couldn't find any relevant information in your knowledge base for this question." 
                        
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
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    
                    try:
                        # 1. Route
                        ftype = FileRouter.route(tmp_path)
                        st.info(f"Detected file type: {ftype.value}")
                        
                        # 2. Parse
                        chunks = []
                        if ftype == FileType.CODE:
                            parser = ASTParser()
                            chunks = parser.parse(tmp_path)
                        elif ftype == FileType.DOCUMENT:
                            parser = DoclingParser()
                            chunks = parser.parse(tmp_path)
                        else:
                            st.warning("Unsupported file type.")
                        
                        st.write(f"Parsed {len(chunks)} chunks.")
                        
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
                                    response = await async_client.generate(
                                        model=ai.model, 
                                        prompt=f"Summarize the following technical content concisely:\n\n{full_text[:4000]}"
                                    )
                                    return response["response"]
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
                        nodes = graph_data.get('nodes', [])
                        edges = graph_data.get('edges', [])
                        st.write(f"Extracted {len(nodes)} nodes and {len(edges)} edges.")
                        
                        keywords = [n['name'] for n in nodes]
                        
                        # 4. Storage
                        st.text("Storing data...")
                        
                        file_meta = db.upload_file_metadata(
                            project_id=st.session_state.selected_project_id,
                            name=uploaded_file.name,
                            path=uploaded_file.name,
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
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
    
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
