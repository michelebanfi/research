
import os
import streamlit as st
import asyncio
import tempfile
from pathlib import Path
from dotenv import load_dotenv

from streamlit_agraph import agraph, Node, Edge, Config as AgraphConfig
from src.database import DatabaseClient
from src.ingestion import FileRouter, DoclingParser, ASTParser, FileType
from src.ai_engine import AIEngine

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Local-Brain Research Assistant", layout="wide")

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

st.title("Local-Brain, Cloud-Memory Research Assistant")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Project Management", "Ingestion", "Knowledge Explorer"])

# --- Project Management ---
if page == "Project Management":
    st.header("Project Management")
    
    with st.expander("Create New Project"):
        new_project_name = st.text_input("Project Name")
        if st.button("Create Project"):
            if db and new_project_name:
                res = db.create_project(new_project_name)
                if res:
                    st.success(f"Project '{new_project_name}' created!")
                else:
                    st.error("Failed to create project.")
            elif not db:
                st.error("Database not connected.")

    st.subheader("Existing Projects")
    if db:
        projects = db.get_projects()
        if projects:
            for p in projects:
                st.write(f"- **{p['name']}** (ID: `{p['id']}`)")
        else:
            st.info("No projects found.")

# --- Ingestion ---
elif page == "Ingestion":
    st.header("Ingestion Pipeline")
    
    if db:
        projects = db.get_projects()
        project_options = {p['name']: p['id'] for p in projects}
        
        if not project_options:
            st.warning("Please create a project first.")
        else:
            selected_project_name = st.selectbox("Select Project", list(project_options.keys()))
            selected_project_id = project_options[selected_project_name]
            
            uploaded_file = st.file_uploader("Upload Document or Code", type=['pdf', 'md', 'txt', 'py', 'docx'])
            
            if uploaded_file and st.button("Process & Ingest"):
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
                        st.text("Running AI Analysis (Summary, Graph, Embeddings) in parallel...")
                        
                        full_text = "\n".join([c['content'] for c in chunks])
                        
                        async def run_ingestion_pipeline():
                            # Create tasks
                            summary_task = ai.generate_summary_async(full_text)
                            graph_task = ai.extract_metadata_graph_async(full_text)
                            
                            # Embedding tasks
                            embedding_tasks = [ai.generate_embedding_async(c['content']) for c in chunks]
                            
                            # Execute all
                            # We can gather summary and graph, and also gather embeddings
                            
                            # Let's group them:
                            # 1. Summary
                            # 2. Graph
                            # 3. Embeddings (list of tasks)
                            
                            results = await asyncio.gather(
                                summary_task,
                                graph_task,
                                asyncio.gather(*embedding_tasks)
                            )
                            return results

                        # Run the pipeline
                        summary, graph_data, embeddings = asyncio.run(run_ingestion_pipeline())
                        
                        # Assign embeddings back to chunks
                        for i, emb in enumerate(embeddings):
                            chunks[i]['embedding'] = emb

                        st.write(f"Generated {len(embeddings)} embeddings.")
                        
                        st.text_area("Summary", summary, height=100)
                        
                        # Graph Data
                        nodes = graph_data.get('nodes', [])
                        edges = graph_data.get('edges', [])
                        
                        st.write(f"Extracted {len(nodes)} nodes and {len(edges)} edges.")
                        
                        # Backward compat: keywords = node names
                        keywords = [n['name'] for n in nodes]
                        if not keywords:
                             # Fallback or empty
                             pass
                        
                        st.write(f"Keywords (Nodes): {', '.join(keywords)}")

                        # 4. Storage
                        st.text("Storing data...")
                        
                        # Store File Metadata first to get ID
                        file_meta = db.upload_file_metadata(
                            project_id=selected_project_id,
                            name=uploaded_file.name,
                            path=uploaded_file.name,
                            summary=summary,
                            metadata={"keywords": keywords}
                        )
                        
                        if file_meta:
                            file_id = file_meta['id']

                            # Store chunks
                            db.store_chunks(file_id, chunks)
                            
                            # Store Keywords (Legacy/Compat)
                            db.store_keywords(file_id, keywords)
                            
                            # Store Graph Data
                            db.store_graph_data(file_id, nodes, edges)
                            
                            st.success("Ingestion Complete!")
                        else:
                            st.error("Failed to upload file metadata.")
                            
                    except Exception as e:
                        st.error(f"Error during ingestion: {e}")
                        import traceback
                        st.text(traceback.format_exc())
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)

# --- Knowledge Explorer ---
elif page == "Knowledge Explorer":
    st.header("Knowledge Explorer")
    
    if db:
        projects = db.get_projects()
        project_options = {p['name']: p['id'] for p in projects}
        
        if project_options:
            selected_project_name = st.selectbox("Select Project", list(project_options.keys()))
            selected_project_id = project_options[selected_project_name]
            
            tab1, tab2, tab3 = st.tabs(["Browse Files", "Semantic Search", "Graph Explorer"])

            with tab1:
                st.subheader("Files")
                files = db.get_project_files(selected_project_id)
                if files:
                    for f in files:
                        with st.expander(f"{f['name']} ({f['processed_at']})"):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"**Summary:** {f['summary']}")
                                st.write(f"**Keywords:** {', '.join(f['metadata'].get('keywords', []))}")
                                
                                # Related Files
                                related = db.get_related_files(f['id'])
                                if related:
                                    st.write("**Related Files:**")
                                    for rf in related:
                                        st.caption(f"- {rf['name']} (Shared Keywords: {rf['shared_count']})")
                                else:
                                    st.caption("No related files found.")

                            with col2:
                                if st.button("Delete File", key=f"del_{f['id']}"):
                                    if db.delete_file(f['id']):
                                        st.success("Deleted!")
                                        st.rerun()
                                    else:
                                        st.error("Delete failed.")
                else:
                    st.info("No files in this project.")
            
            with tab2:
                st.subheader("Semantic Search with Re-ranking")
                query = st.text_input("Ask a question or search...")
                do_rerank = st.checkbox("Enable Re-ranking (slower but more accurate)")
                
                if query and st.button("Search"):
                    with st.spinner("Searching..."):
                        query_embedding = ai.generate_embedding(query)
                        if query_embedding:
                            # 1. Vector Search
                            results = db.search_vectors(query_embedding, match_threshold=0.3, project_id=selected_project_id)
                            
                            # 2. Re-ranking
                            if do_rerank and results:
                                with st.spinner("Re-ranking results with Qwen..."):
                                    results = ai.rerank_results(query, results, top_k=5)
                            
                            # --- Hybrid Search Context ---
                            if results:
                                with st.expander("📚 Broaden Your Research (Graph Context)", expanded=True):
                                    top_file_id = results[0].get('file_id')
                                    if top_file_id:
                                        col_g1, col_g2 = st.columns(2)
                                        
                                        with col_g1:
                                            st.markdown("**Related Concepts (from Top Result):**")
                                            graph_rows = db.get_file_graph(top_file_id)
                                            concepts = set()
                                            for row in graph_rows:
                                                # Filter out Persons or strict types if available
                                                if row.get('source_type') not in ['Person', 'Organization'] and row.get('target_type') not in ['Person', 'Organization']:
                                                    concepts.add(row['source_name'])
                                                    concepts.add(row['target_name'])
                                            
                                            if concepts:
                                                st.info(", ".join(list(concepts)[:10]) + ("..." if len(concepts)>10 else ""))
                                            else:
                                                st.caption("No graph concepts found.")

                                        with col_g2:
                                            st.markdown("**Related Papers (via Graph):**")
                                            related_files = db.get_related_files(top_file_id)
                                            if related_files:
                                                for rf in related_files:
                                                    st.markdown(f"- **{rf['name']}** ({rf['shared_count']} shared concepts)")
                                            else:
                                                st.caption("No related papers found.")

                            for res in results:
                                score = res.get('rerank_score', res.get('similarity'))
                                score_label = "Re-rank Score" if 'rerank_score' in res else "Sim"
                                st.markdown(f"**Chunk from file {res.get('file_id')}** ({score_label}: {score:.2f})")
                                st.markdown(f"> {res.get('content')}")
                                st.divider()
                        else:
                            st.error("Failed to generate query embedding.")
            
            with tab3:
                st.subheader("Graph Explorer")
                
                # Persistence logic using session state
                if "graph_nodes" not in st.session_state:
                    st.session_state.graph_nodes = []
                if "graph_edges" not in st.session_state:
                    st.session_state.graph_edges = []
                
                st.caption("Visualizing the Global Knowledge Graph for the entire project.")
                
                if st.button("Load/Refresh Graph"):
                    with st.spinner("Loading Global Graph Data..."):
                        # Always get Global Project Graph
                        graph_data_rows = db.get_project_graph(selected_project_id, limit=500)
                        
                        # Process Data into Nodes/Edges
                        nodes = []
                        edges = []
                        added_nodes = set()
                        
                        for row in graph_data_rows:
                            s_name = row['source_name']
                            t_name = row['target_name']
                            s_type = row['source_type']
                            t_type = row['target_type']
                            relation = row['edge_type']
                            
                            # Simple logic for colors
                            if s_name not in added_nodes:
                                color = "#FF5733" if s_type == "Concept" else "#33C1FF"
                                nodes.append(Node(id=s_name, label=s_name, size=20, color=color))
                                added_nodes.add(s_name)
                            
                            if t_name not in added_nodes:
                                color = "#FF5733" if t_type == "Concept" else "#33C1FF"
                                nodes.append(Node(id=t_name, label=t_name, size=20, color=color))
                                added_nodes.add(t_name)
                                
                            edges.append(Edge(source=s_name, target=t_name, label=relation))
                        
                        # Update Session State
                        st.session_state.graph_nodes = nodes
                        st.session_state.graph_edges = edges
                
                # Always render if data exists in session state
                if st.session_state.graph_nodes:
                    st.success(f"Visualizing {len(st.session_state.graph_nodes)} nodes and {len(st.session_state.graph_edges)} edges.")
                    
                    config = AgraphConfig(
                        width=800, 
                        height=600, 
                        directed=True, 
                        nodeHighlightBehavior=True, 
                        highlightColor="#F7A7A6",
                        collapsible=True,
                        physics=True
                    )
                    
                    clicked_node = agraph(nodes=st.session_state.graph_nodes, edges=st.session_state.graph_edges, config=config)
                    
                    if clicked_node:
                        st.info(f"You clicked: {clicked_node}")
                else:
                    st.info("Click 'Load/Refresh Graph' to visualize the knowledge base.")


