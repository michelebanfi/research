# Research Assistant - React + FastAPI Migration

## Overview

This is the new React + FastAPI version of the Research Assistant, replacing the Streamlit interface with a modern, customizable React frontend.

## Architecture

```
research/
├── backend/              # FastAPI Python backend
│   ├── app/
│   │   └── main.py       # Main FastAPI app with WebSocket
│   ├── logs/             # Backend logs
│   └── requirements.txt  # Python dependencies
├── frontend/             # React + TypeScript frontend
│   ├── src/
│   │   ├── components/   # React components
│   │   ├── stores/       # Zustand state management
│   │   └── services/     # API client
│   └── package.json
├── src/                  # Existing Python modules (reused)
├── start_backend.sh      # Start backend only
├── start_frontend.sh     # Start frontend only
└── start_all.sh          # Start both

```

## Quick Start

### Option 1: Start Everything at Once

```bash
./start_all.sh
```

This will start both the backend (port 8000) and frontend (port 5173).

### Option 2: Start Separately (for development)

Terminal 1 - Backend:
```bash
./start_backend.sh
```

Terminal 2 - Frontend:
```bash
./start_frontend.sh
```

## URLs

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Features

### Chat Tab
- Real-time chat with WebSocket streaming
- Live process monitor showing agent reasoning
- Context panel with retrieved sources
- Settings: Re-ranking toggle, context chunks slider, reasoning mode

### Ingest Tab
- File upload (PDF, MD, TXT, PY, DOCX)
- View existing files with metadata
- Delete files

### Graph Tab
- Interactive knowledge graph visualization using React Flow
- Color-coded nodes by type (Concept, Tool, System, Metric, Person)
- Click nodes to see details
- Legend and controls

## State Management

Uses Zustand for simple, effective state management:
- Projects list and selection
- Chat history
- Agent events (for live process monitor)
- Retrieved context and matched concepts
- Settings (rerank, top_k, reasoning mode)

## API Endpoints

### REST Endpoints
- `GET /api/projects` - List all projects
- `POST /api/projects` - Create new project
- `GET /api/projects/{id}/files` - Get project files
- `DELETE /api/files/{id}` - Delete a file
- `POST /api/upload` - Upload and process a file
- `GET /api/projects/{id}/graph` - Get knowledge graph data

### WebSocket
- `WS /ws/chat` - Real-time chat with streaming events

## Logging

Backend logs are written to:
- Console (stdout)
- `backend/logs/app.log`

## Migration from Streamlit

### What Changed
1. **UI Framework**: Streamlit → React + TypeScript + Tailwind CSS
2. **Backend**: Direct Python execution → FastAPI with async endpoints
3. **Real-time Updates**: Streamlit session state → WebSocket streaming
4. **State Management**: Streamlit session state → Zustand stores
5. **Graph Viz**: agraph → React Flow

### What Stayed the Same
- All Python business logic (AIEngine, DatabaseClient, Agents)
- Database schema and Supabase integration
- File processing and ingestion pipeline
- Knowledge graph extraction

## Development

### Backend Development
The backend reuses all existing Python modules from `src/`. FastAPI imports and uses them directly, so any improvements to the core logic benefit both the old and new interfaces.

### Frontend Development
Frontend is built with:
- React 18 + TypeScript
- Vite (fast dev server)
- Tailwind CSS (styling)
- Zustand (state management)
- React Flow (graph visualization)
- Lucide React (icons)
- Axios (HTTP client)

## Troubleshooting

### Backend won't start
1. Check `.env` file has required variables (SUPABASE_URL, SUPABASE_KEY, etc.)
2. Ensure virtual environment is activated: `source .venv/bin/activate`
3. Check logs: `tail -f backend/logs/app.log`

### Frontend won't start
1. Ensure dependencies are installed: `cd frontend && npm install`
2. Check if port 5173 is available
3. Check browser console for errors

### WebSocket connection fails
1. Ensure backend is running on port 8000
2. Check browser console for connection errors
3. Verify CORS settings in `backend/app/main.py`

## Next Steps / TODO

- [ ] Add Agent Graph visualization (Mermaid diagram)
- [ ] Add file preview functionality
- [ ] Implement search within context
- [ ] Add export functionality
- [ ] Add keyboard shortcuts
- [ ] Optimize graph layout algorithm
- [ ] Add dark/light mode toggle
