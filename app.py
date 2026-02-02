
import os
import streamlit as st
import tempfile
from pathlib import Path
from dotenv import load_dotenv

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
                        
                        # 3. AI Processing
                        full_text = "\n".join([c['content'] for c in chunks])
                        summary = ai.generate_summary(full_text)
                        keywords = ai.extract_keywords(full_text)
                        
                        st.text_area("Summary", summary, height=100)
                        st.write(f"Keywords: {', '.join(keywords)}")
                        
                        # 4. Embeddings & Storage
                        st.text("Generating embeddings and storing...")
                        
                        # Store File Metadata
                        file_meta = db.upload_file_metadata(
                            project_id=selected_project_id,
                            name=uploaded_file.name,
                            path=uploaded_file.name,
                            summary=summary,
                            metadata={"keywords": keywords}
                        )
                        
                        if file_meta:
                            file_id = file_meta['id']
                            
                            # Embed chunks
                            for chunk in chunks:
                                chunk['embedding'] = ai.generate_embedding(chunk['content'])
                            
                            # Store chunks
                            db.store_chunks(file_id, chunks)
                            
                            # Store Keywords and Links
                            db.store_keywords(file_id, keywords)
                            
                            st.success("Ingestion Complete!")
                        else:
                            st.error("Failed to upload file metadata.")
                            
                    except Exception as e:
                        st.error(f"Error during ingestion: {e}")
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
            
            tab1, tab2 = st.tabs(["Browse Files", "Semantic Search"])

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
                            results = db.search_vectors(query_embedding, match_threshold=0.3, match_count=10, project_id=selected_project_id)
                            
                            # 2. Re-ranking
                            if do_rerank and results:
                                with st.spinner("Re-ranking results with Qwen..."):
                                    results = ai.rerank_results(query, results, top_k=5)
                            
                            for res in results:
                                score = res.get('rerank_score', res.get('similarity'))
                                score_label = "Re-rank Score" if 'rerank_score' in res else "Sim"
                                st.markdown(f"**Chunk from file {res.get('file_id')}** ({score_label}: {score:.2f})")
                                st.markdown(f"> {res.get('content')}")
                                st.divider()
                        else:
                            st.error("Failed to generate query embedding.")
