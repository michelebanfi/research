import os
import streamlit as st
import asyncio
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Fix event loop issues in Streamlit
from src.utils.async_utils import get_or_create_event_loop
# Ensure loop exists before unrelated async inits
loop = get_or_create_event_loop()

from src.database import DatabaseClient
from src.ai_engine import AIEngine
from src.config import Config

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
def get_shared_ranker():
    """Cache the heavy Re-ranker model."""
    from flashrank import Ranker
    return Ranker(model_name=Config.RERANK_MODEL_NAME, cache_dir="./.cache")

def get_ai_engine():
    """
    Ephemerally create AIEngine to bind asyncio objects (Semaphores) to current loop.
    Fail-safe against Streamlit threading model.
    Pass cached ranker to avoid reloading heavy model.
    """
    ranker = get_shared_ranker()
    return AIEngine(ranker=ranker)

db = get_db_client()
ai = get_ai_engine() # No longer cached, but lightweight

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
    st.session_state.matched_concepts = []
if "chats" not in st.session_state:
    st.session_state.chats = {}  # {chat_id: [{"role": "user", "content": "..."}]}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

def load_logo():
    """Load and return the ASCII logo."""
    logo_path = Path(__file__).parent / "logo.txt"
    if logo_path.exists():
        with open(logo_path, "r") as f:
            return f.read()
    return None

def start_new_chat():
    """Start a new chat session."""
    import uuid
    new_id = str(uuid.uuid4())
    st.session_state.chats[new_id] = []
    st.session_state.current_chat_id = new_id
    st.session_state.chat_history = []
    st.session_state.last_context = []
    st.session_state.matched_concepts = []
    st.rerun()

def switch_chat(chat_id: str):
    """Switch to an existing chat."""
    st.session_state.current_chat_id = chat_id
    st.session_state.chat_history = st.session_state.chats.get(chat_id, [])
    st.rerun()

# Handle new chat trigger from UI
if st.session_state.get("_trigger_new_chat"):
    st.session_state._trigger_new_chat = False
    start_new_chat()

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
                st.session_state.chats = {}
                st.session_state.current_chat_id = None
                st.rerun()
            
            st.caption(f"ID: `{st.session_state.selected_project_id}`")
        
            # Chat history selector
            st.markdown("---")
            st.header("💬 Chats")
            
            if st.session_state.chats:
                chat_options = list(st.session_state.chats.keys())
                chat_labels = [f"Chat {i+1}" for i in range(len(chat_options))]
                chat_map = dict(zip(chat_labels, chat_options))
                
                current_label = "New Chat"
                if st.session_state.current_chat_id:
                    idx = chat_options.index(st.session_state.current_chat_id) if st.session_state.current_chat_id in chat_options else -1
                    if idx >= 0:
                        current_label = chat_labels[idx]
                
                selected_label = st.selectbox(
                    "Select Chat",
                    options=["➕ New Chat"] + chat_labels,
                    index=0 if current_label == "New Chat" else chat_labels.index(current_label) + 1,
                    key="chat_selector"
                )
                
                if selected_label == "➕ New Chat":
                    start_new_chat()
                elif selected_label in chat_map:
                    switch_chat(chat_map[selected_label])
                
                # Show chat count
                st.caption(f"{len(st.session_state.chats)} chat(s)")
            else:
                if st.button("➕ Start First Chat", use_container_width=True):
                    start_new_chat()
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
    tab_chat, tab_ingest, tab_graph, tab_arch = st.tabs([
        "💬 Chat", "📥 Ingest Files", "🕸️ Knowledge Graph", "🏗️ Agent Graph"
    ])
    
    # Imports for UI modules
    from src.ui.chat import render_chat_tab
    from src.ui.ingest import render_ingest_tab
    from src.ui.graph import render_graph_tab
    from src.ui.graph_view import render_graph_tab as render_agent_graph_tab
    
    with tab_chat:
        render_chat_tab(db, ai)
    
    with tab_ingest:
        render_ingest_tab(db, ai)
        
    with tab_graph:
        render_graph_tab(db)

    with tab_arch:
        render_agent_graph_tab()

else:
    st.info("👈 Please select or create a project from the sidebar to get started.")
