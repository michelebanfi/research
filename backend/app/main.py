"""
FastAPI Backend for Research Assistant
Simple, effective, with comprehensive logging
"""

import os
import sys
import json
import logging
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('backend/logs/app.log')
    ]
)
logger = logging.getLogger(__name__)

# Import existing modules (they'll be in Python path when we run from root)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import after adding to path
from src.database import DatabaseClient
from src.ai_engine import AIEngine
from src.agent import ResearchAgent
from src.agent import AgentResponse
from src.models import ReasoningResponse
from src.config import Config
from flashrank import Ranker

# Global state (simple, effective for single-user)
class AppState:
    def __init__(self):
        self.db: Optional[DatabaseClient] = None
        self.ranker: Optional[Ranker] = None
        self.ai_engine: Optional[AIEngine] = None
        self.ingestion_progress: Dict[str, Dict] = {}  # file_id -> progress info
        logger.info("AppState initialized")

class IngestionProgress(BaseModel):
    file_id: str
    stage: str
    progress: float  # 0-100
    message: str
    chunks_count: int = 0
    nodes_count: int = 0
    edges_count: int = 0

app_state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("🚀 Starting up FastAPI backend...")
    
    # Initialize database
    try:
        app_state.db = DatabaseClient()
        logger.info("✅ Database client initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        raise
    
    # Initialize ranker (heavy model, load once)
    try:
        logger.info("Loading FlashRank model...")
        app_state.ranker = Ranker(
            model_name=Config.RERANK_MODEL_NAME,
            cache_dir="./.cache"
        )
        logger.info("✅ Ranker loaded")
    except Exception as e:
        logger.error(f"❌ Failed to load ranker: {e}")
        raise
    
    # Initialize AI Engine
    try:
        app_state.ai_engine = AIEngine(ranker=app_state.ranker)
        logger.info("✅ AI Engine initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize AI Engine: {e}")
        raise
    
    logger.info("✅ Backend startup complete!")
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down backend...")

app = FastAPI(
    title="Research Assistant API",
    description="Backend for the Research Assistant application",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite and React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Pydantic Models for API
# ============================================================================

class ProjectCreate(BaseModel):
    name: str

class ProjectResponse(BaseModel):
    id: str
    name: str
    created_at: str

class ChatMessage(BaseModel):
    role: str
    content: str
    model: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    project_id: str
    chat_history: List[ChatMessage] = []
    do_rerank: bool = True
    top_k: int = 5
    reasoning_mode: bool = False

class ChatResponse(BaseModel):
    success: bool
    answer: str
    model_name: str
    retrieved_chunks: List[Dict[str, Any]] = []
    matched_concepts: List[str] = []
    error: Optional[str] = None

class FileUploadResponse(BaseModel):
    success: bool
    file_id: Optional[str] = None
    temp_file_id: Optional[str] = None  # For progress tracking
    message: str
    chunks_count: int = 0
    nodes_count: int = 0
    edges_count: int = 0

# ============================================================================
# REST Endpoints
# ============================================================================

@app.get("/")
async def root():
    logger.info("Health check request")
    return {"status": "ok", "message": "Research Assistant API is running"}

@app.get("/api/projects", response_model=List[ProjectResponse])
async def get_projects():
    """Get all projects"""
    logger.info("Fetching all projects")
    try:
        projects = app_state.db.get_projects()
        logger.info(f"Found {len(projects)} projects")
        return projects
    except Exception as e:
        logger.error(f"Error fetching projects: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/projects", response_model=ProjectResponse)
async def create_project(project: ProjectCreate):
    """Create a new project"""
    logger.info(f"Creating project: {project.name}")
    try:
        result = app_state.db.create_project(project.name)
        if result:
            logger.info(f"Project created: {result['id']}")
            return result
        else:
            raise HTTPException(status_code=500, detail="Failed to create project")
    except Exception as e:
        logger.error(f"Error creating project: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/projects/{project_id}/files")
async def get_project_files(project_id: str):
    """Get all files for a project"""
    logger.info(f"Fetching files for project: {project_id}")
    try:
        files = app_state.db.get_project_files(project_id)
        logger.info(f"Found {len(files)} files")
        return files
    except Exception as e:
        logger.error(f"Error fetching files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/files/{file_id}")
async def delete_file(file_id: str):
    """Delete a file"""
    logger.info(f"Deleting file: {file_id}")
    try:
        success = app_state.db.delete_file(file_id)
        if success:
            return {"success": True}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete file")
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/upload/progress/{temp_file_id}")
async def get_upload_progress(temp_file_id: str):
    """Get the current progress of a file upload"""
    progress = app_state.ingestion_progress.get(temp_file_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Progress not found")
    return progress

@app.get("/api/files/{file_id}/details")
async def get_file_details(file_id: str):
    """Get detailed information about a file including chunks and structure"""
    logger.info(f"Fetching file details: {file_id}")
    try:
        # Get file metadata
        file_response = app_state.db.client.table("files").select("*").eq("id", file_id).execute()
        if not file_response.data:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_data = file_response.data[0]
        
        # Get chunks with full metadata
        chunks = app_state.db.get_file_chunks(file_id)
        
        # Get sections
        sections = app_state.db.get_file_sections(file_id)
        
        # Get keywords (for backward compatibility) and typed entities
        keywords = app_state.db.get_file_keywords(file_id)
        entities = app_state.db.get_file_entities(file_id)
        
        # Calculate statistics
        stats = {
            "total_chunks": len(chunks),
            "table_chunks": len([c for c in chunks if c.get("is_table")]),
            "reference_chunks": len([c for c in chunks if c.get("is_reference")]),
            "parent_chunks": len([c for c in chunks if c.get("chunk_level", 0) == 0]),
            "leaf_chunks": len([c for c in chunks if c.get("chunk_level", 0) > 0]),
        }
        
        return {
            "file": file_data,
            "chunks": chunks,
            "sections": sections,
            "keywords": keywords,
            "entities": entities,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error fetching file details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/projects/{project_id}/graph")
async def get_project_graph(project_id: str, limit: int = 500):
    """Get knowledge graph for a project"""
    logger.info(f"Fetching graph for project: {project_id}")
    try:
        graph_data = app_state.db.get_project_graph(project_id, limit=limit)
        
        # Transform to frontend-friendly format
        nodes = []
        edges = []
        added_nodes = set()
        
        type_colors = {
            "Concept": "#FF6B6B",
            "Tool": "#4ECDC4",
            "System": "#45B7D1",
            "Metric": "#96CEB4",
            "Person": "#DDA0DD",
        }
        
        for row in graph_data:
            s_name = row['source_name']
            t_name = row['target_name']
            s_type = row['source_type']
            t_type = row['target_type']
            relation = row['edge_type']
            
            if s_name not in added_nodes:
                nodes.append({
                    "id": s_name,
                    "label": s_name,
                    "type": s_type,
                    "color": type_colors.get(s_type, "#95A5A6")
                })
                added_nodes.add(s_name)
            
            if t_name not in added_nodes:
                nodes.append({
                    "id": t_name,
                    "label": t_name,
                    "type": t_type,
                    "color": type_colors.get(t_type, "#95A5A6")
                })
                added_nodes.add(t_name)
            
            edges.append({
                "source": s_name,
                "target": t_name,
                "label": relation
            })
        
        logger.info(f"Graph: {len(nodes)} nodes, {len(edges)} edges")
        return {"nodes": nodes, "edges": edges}
    except Exception as e:
        logger.error(f"Error fetching graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# WebSocket for Chat Streaming
# ============================================================================

class ConnectionManager:
    """Connection manager for WebSockets with safe sending"""
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_message(self, message: dict, websocket: WebSocket) -> bool:
        """Safely send a message, return False if failed"""
        try:
            if websocket.client_state.CONNECTED:
                await websocket.send_json(message)
                return True
        except Exception as e:
            logger.debug(f"Failed to send message: {e}")
        return False

manager = ConnectionManager()

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for streaming chat"""
    await manager.connect(websocket)
    logger.info("Chat WebSocket connected")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            logger.info(f"Received chat request for project: {data.get('project_id')}")
            
            # Parse request
            message = data.get('message', '')
            project_id = data.get('project_id')
            chat_history = data.get('chat_history', [])
            do_rerank = data.get('do_rerank', True)
            reasoning_mode = data.get('reasoning_mode', False)
            
            if not message or not project_id:
                success = await manager.send_message({
                    "type": "error",
                    "content": "Missing message or project_id"
                }, websocket)
                if not success:
                    logger.warning("Could not send error message, client disconnected")
                    break
                continue
            
            # Convert chat history format
            history = [{"role": msg["role"], "content": msg["content"]} for msg in chat_history]
            
            # Create agent with event callback
            agent_events = []
            
            async def safe_send_event(event: dict):
                """Safely send event to frontend"""
                try:
                    if websocket.client_state.CONNECTED:
                        await websocket.send_json({
                            "type": "event",
                            "event": event
                        })
                except Exception as e:
                    logger.debug(f"Could not send event (connection likely closed): {e}")
            
            def event_callback(event: dict):
                """Send events to frontend in real-time"""
                agent_events.append(event)
                # Schedule sending without waiting
                asyncio.create_task(safe_send_event(event))
            
            try:
                # Create agent
                agent = ResearchAgent(
                    ai_engine=app_state.ai_engine,
                    database=app_state.db,
                    project_id=project_id,
                    do_rerank=do_rerank,
                    event_callback=event_callback
                )
                
                # Run agent
                await manager.send_message({
                    "type": "status",
                    "content": "🤔 Thinking..."
                }, websocket)
                
                result = await agent.run(
                    user_query=message,
                    chat_history=history,
                    reasoning_mode=reasoning_mode
                )
                
                # Send final result
                if isinstance(result, ReasoningResponse):
                    await manager.send_message({
                        "type": "result",
                        "content": {
                            "success": result.success,
                            "answer": result.final_output if result.success else result.error,
                            "model_name": result.model_name,
                            "reasoning_mode": True,
                            "plan": {
                                "context_needed": result.plan.context_needed if result.plan else "",
                                "goal": result.plan.goal if result.plan else "",
                                "verification_logic": result.plan.verification_logic if result.plan else ""
                            } if result.plan else None,
                            "attempts": len(result.attempts)
                        }
                    }, websocket)
                else:
                    await manager.send_message({
                        "type": "result",
                        "content": {
                            "success": True,
                            "answer": result.answer,
                            "model_name": result.model_name,
                            "reasoning_mode": False,
                            "retrieved_chunks": result.retrieved_chunks,
                            "matched_concepts": result.matched_concepts
                        }
                    }, websocket)

                logger.info(f"Chat completed. Model: {result.model_name}")

            except Exception as e:
                logger.error(f"Error in chat: {e}", exc_info=True)
                await manager.send_message({
                    "type": "error",
                    "content": f"Error: {str(e)}"
                }, websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Chat WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        manager.disconnect(websocket)

# ============================================================================
# File Upload Endpoint
# ============================================================================

def update_progress(file_id: str, stage: str, progress: float, message: str, chunks: int = 0, nodes: int = 0, edges: int = 0):
    """Update ingestion progress in global state"""
    app_state.ingestion_progress[file_id] = {
        "file_id": file_id,
        "stage": stage,
        "progress": progress,
        "message": message,
        "chunks_count": chunks,
        "nodes_count": nodes,
        "edges_count": edges,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    project_id: str = Form(...)
):
    """Upload and process a file with progress tracking"""
    logger.info(f"Upload request: {file.filename} for project {project_id}")
    
    # Generate temporary file_id for progress tracking
    temp_file_id = f"temp_{datetime.now().timestamp()}"
    
    try:
        # Stage 1: Saving file (0-10%)
        update_progress(temp_file_id, "saving", 5, "Saving uploaded file...")
        upload_dir = Path("static/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / file.filename
        
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        logger.info(f"File saved: {file_path}")
        
        # Stage 2: Parsing (10-30%)
        update_progress(temp_file_id, "parsing", 15, "Parsing document structure...")
        from src.ingestion import FileRouter, DoclingParser, ASTParser, FileType
        
        # Detect file type
        ftype = FileRouter.route(str(file_path))
        logger.info(f"Detected file type: {ftype.value}")
        
        # Parse file
        chunks = []
        code_graph_data = {"nodes": [], "edges": []}
        
        if ftype == FileType.CODE:
            parser = ASTParser()
            parse_result = parser.parse(str(file_path))
            chunks = parse_result.get("chunks", [])
            code_graph_data = parse_result.get("graph_data", {"nodes": [], "edges": []})
        elif ftype == FileType.DOCUMENT:
            parser = DoclingParser()
            chunks = parser.parse(str(file_path))
        else:
            return FileUploadResponse(
                success=False,
                message="Unsupported file type"
            )
        
        update_progress(temp_file_id, "parsing", 30, f"Parsed {len(chunks)} chunks", chunks=len(chunks))
        logger.info(f"Parsed {len(chunks)} chunks")
        
        # AI Processing
        full_text = "\n".join([c['content'] for c in chunks])
        
        async def run_ingestion_pipeline():
            import ollama
            async_client = ollama.AsyncClient()
            
            # Stage 3: Generating summary (30-50%)
            update_progress(temp_file_id, "summarizing", 40, "Generating document summary...", chunks=len(chunks))
            summary = await app_state.ai_engine.generate_summary_async(full_text[:4000])
            
            # Stage 4: Extracting knowledge graph (50-70%)
            update_progress(temp_file_id, "extracting_graph", 60, "Extracting entities and relationships...", chunks=len(chunks))
            graph_data = await app_state.ai_engine.extract_metadata_graph_async(full_text)
            
            # Stage 5: Generating embeddings (70-90%)
            update_progress(temp_file_id, "embedding", 75, "Generating embeddings for chunks...", chunks=len(chunks))
            embeddings = []
            total_batches = (len(chunks) + 9) // 10
            for batch_idx in range(0, len(chunks), 10):  # Batch size 10
                batch_num = batch_idx // 10 + 1
                batch_progress = 75 + (batch_num / total_batches) * 15
                update_progress(temp_file_id, "embedding", batch_progress, 
                              f"Generating embeddings (batch {batch_num}/{total_batches})...", chunks=len(chunks))
                
                batch = chunks[batch_idx:batch_idx+10]
                batch_tasks = []
                for chunk in batch:
                    text = chunk.get('embedding_text', chunk['content'])
                    batch_tasks.append(async_client.embeddings(
                        model=app_state.ai_engine.embed_model,
                        prompt=text
                    ))
                
                try:
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    for res in batch_results:
                        if isinstance(res, Exception):
                            embeddings.append(None)
                        else:
                            embeddings.append(res["embedding"])
                except Exception as e:
                    logger.error(f"Embedding batch error: {e}")
                    embeddings.extend([None] * len(batch))
            
            return summary, graph_data, embeddings
        
        # Run pipeline
        from src.utils.async_utils import run_sync
        summary, graph_data, embeddings = run_sync(run_ingestion_pipeline())
        
        # Attach embeddings
        valid_embeddings = 0
        for i, emb in enumerate(embeddings):
            if emb and len(emb) > 0:
                chunks[i]['embedding'] = emb
                valid_embeddings += 1
            else:
                chunks[i]['embedding'] = None
        
        logger.info(f"Generated {valid_embeddings} embeddings")
        
        # Extract keywords
        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('edges', [])
        
        if code_graph_data["nodes"]:
            nodes.extend(code_graph_data["nodes"])
            edges.extend(code_graph_data["edges"])
        
        keywords = [n['name'] for n in nodes]
        
        # Stage 6: Storing in database (90-100%)
        update_progress(temp_file_id, "storing", 95, "Storing data in database...", 
                       chunks=len(chunks), nodes=len(nodes), edges=len(edges))
        
        # Store in database
        file_meta = app_state.db.upload_file_metadata(
            project_id=project_id,
            name=file.filename,
            path=str(file_path),
            summary=summary,
            metadata={"keywords": keywords}
        )
        
        if file_meta:
            file_id = file_meta['id']
            
            # Extract and store sections
            sections_data = []
            seen_paths = set()
            
            for c in chunks:
                headings = c['metadata'].get('headings', [])
                if not headings:
                    continue
                
                current_path = []
                for title in headings:
                    current_path.append(title)
                    path_tuple = tuple(current_path)
                    
                    if path_tuple not in seen_paths:
                        seen_paths.add(path_tuple)
                        sections_data.append({
                            'title': title,
                            'level': len(current_path),
                            'path': current_path,
                            'parent_path': current_path[:-1] if len(current_path) > 1 else []
                        })
            
            section_map = {}
            if sections_data:
                section_map = app_state.db.store_sections(file_id, sections_data)
            
            # Link chunks to sections
            for c in chunks:
                headings = c['metadata'].get('headings', [])
                if headings:
                    path_tuple = tuple(headings)
                    if path_tuple in section_map:
                        c['section_id'] = section_map[path_tuple]
            
            # Store everything
            app_state.db.store_chunks(file_id, chunks)
            app_state.db.store_keywords(file_id, keywords)
            app_state.db.store_graph_data(file_id, nodes, edges)
            
            # Final progress update
            update_progress(temp_file_id, "complete", 100, "Ingestion complete!", 
                           chunks=len(chunks), nodes=len(nodes), edges=len(edges))
            
            logger.info(f"Ingestion complete: {file_id}")
            
            return FileUploadResponse(
                success=True,
                file_id=file_id,
                temp_file_id=temp_file_id,
                message="File ingested successfully",
                chunks_count=len(chunks),
                nodes_count=len(nodes),
                edges_count=len(edges)
            )
        else:
            return FileUploadResponse(
                success=False,
                message="Failed to store file metadata"
            )
            
    except Exception as e:
        logger.error(f"Error uploading file: {e}", exc_info=True)
        return FileUploadResponse(
            success=False,
            message=f"Error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
