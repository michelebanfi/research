"""
Local JSON-file persistence for chat conversations and paper analysis results.

Directory layout under DATA_DIR (.data/):
  chats/{project_id}/{chat_id}.json
  analyses/{project_id}/{file_id}.json
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / ".data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, data: dict) -> None:
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Chats
# ---------------------------------------------------------------------------

def _chats_dir(project_id: str) -> Path:
    return DATA_DIR / "chats" / project_id


def save_chat(project_id: str, chat_id: str, data: dict) -> None:
    """Persist (create or update) a chat conversation."""
    path = _chats_dir(project_id) / f"{chat_id}.json"

    # Preserve created_at from existing file if present
    existing = _read_json(path)
    created_at = (existing or {}).get("created_at", datetime.now().isoformat())

    record = {
        "id": chat_id,
        "project_id": project_id,
        "title": data.get("title", ""),
        "created_at": created_at,
        "updated_at": datetime.now().isoformat(),
        "messages": data.get("messages", []),
    }
    _write_json(path, record)
    logger.debug("Saved chat %s for project %s", chat_id, project_id)


def load_chat(project_id: str, chat_id: str) -> Optional[dict]:
    """Load a single chat by ID."""
    return _read_json(_chats_dir(project_id) / f"{chat_id}.json")


def list_chats(project_id: str) -> List[dict]:
    """Return summary list of all chats for a project (sorted by updated_at desc)."""
    d = _chats_dir(project_id)
    if not d.exists():
        return []
    results: List[dict] = []
    for p in d.glob("*.json"):
        data = _read_json(p)
        if data is None:
            continue
        results.append({
            "id": data.get("id", p.stem),
            "title": data.get("title", "Untitled"),
            "created_at": data.get("created_at", ""),
            "updated_at": data.get("updated_at", ""),
            "message_count": len(data.get("messages", [])),
        })
    results.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return results


def delete_chat(project_id: str, chat_id: str) -> bool:
    """Delete a chat file. Returns True if the file existed."""
    path = _chats_dir(project_id) / f"{chat_id}.json"
    if path.exists():
        path.unlink()
        logger.info("Deleted chat %s for project %s", chat_id, project_id)
        return True
    return False


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def _analyses_dir(project_id: str) -> Path:
    return DATA_DIR / "analyses" / project_id


def save_analysis(project_id: str, file_id: str, data: dict) -> None:
    """Persist an analysis result."""
    path = _analyses_dir(project_id) / f"{file_id}.json"

    existing = _read_json(path)
    created_at = (existing or {}).get("created_at", datetime.now().isoformat())

    record = {
        "file_id": file_id,
        "project_id": project_id,
        "file_name": data.get("file_name", ""),
        "created_at": created_at,
        "updated_at": datetime.now().isoformat(),
        "markdown": data.get("markdown", ""),
        "events": data.get("events", []),
    }
    _write_json(path, record)
    logger.debug("Saved analysis for file %s in project %s", file_id, project_id)


def load_analysis(project_id: str, file_id: str) -> Optional[dict]:
    """Load a saved analysis."""
    return _read_json(_analyses_dir(project_id) / f"{file_id}.json")


def list_analyses(project_id: str) -> List[dict]:
    """Return summary list of all saved analyses for a project."""
    d = _analyses_dir(project_id)
    if not d.exists():
        return []
    results: List[dict] = []
    for p in d.glob("*.json"):
        data = _read_json(p)
        if data is None:
            continue
        results.append({
            "file_id": data.get("file_id", p.stem),
            "file_name": data.get("file_name", ""),
            "created_at": data.get("created_at", ""),
            "updated_at": data.get("updated_at", ""),
        })
    results.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return results


def delete_analysis(project_id: str, file_id: str) -> bool:
    """Delete an analysis file. Returns True if the file existed."""
    path = _analyses_dir(project_id) / f"{file_id}.json"
    if path.exists():
        path.unlink()
        logger.info("Deleted analysis for file %s in project %s", file_id, project_id)
        return True
    return False
