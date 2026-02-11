import os
import streamlit as st
import asyncio
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Fix event loop issues in Streamlit
import nest_asyncio
nest_asyncio.apply()

from src.database import DatabaseClient
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
    
    # Imports for UI modules
    from src.ui.chat import render_chat_tab
    from src.ui.ingest import render_ingest_tab
    from src.ui.graph import render_graph_tab
    
    with tab_chat:
        render_chat_tab(db, ai)
    
    with tab_ingest:
        render_ingest_tab(db, ai)
        
    with tab_graph:
        render_graph_tab(db)

else:
    st.info("👈 Please select or create a project from the sidebar to get started.")
