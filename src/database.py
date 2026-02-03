import os
from supabase import create_client, Client
from typing import List, Dict, Any, Optional
from src.config import Config

class DatabaseClient:
    def __init__(self):
        # Validation happens here or handled gracefully
        try:
             Config.validate()
        except ValueError as e:
             # Only raise if we are likely in runtime needing DB
             # But for now, let's allow it to fail fast as before
             raise e
             
        self.url = Config.SUPABASE_URL
        self.key = Config.SUPABASE_KEY
        self.client: Client = create_client(self.url, self.key)

    def create_project(self, name: str) -> Dict[str, Any]:
        """Creates a new project and returns the inserted row."""
        try:
            data, count = self.client.table("projects").insert({"name": name}).execute()
            if data and len(data) > 0:
                 return data[1][0]
            # Fallback for old supabase-py version
            if hasattr(data, 'data') and data.data:
                return data.data[0]
            return None
        except Exception as e:
            print(f"Error creating project: {e}")
            return None

    def get_projects(self) -> List[Dict[str, Any]]:
        """Retrieves all projects."""
        try:
            response = self.client.table("projects").select("*").order("created_at", desc=True).execute()
            return response.data
        except Exception as e:
            print(f"Error getting projects: {e}")
            return []

    def get_project_files(self, project_id: str) -> List[Dict[str, Any]]:
        """Retrieves all files for a specific project."""
        try:
            response = self.client.table("files").select("*").eq("project_id", project_id).order("processed_at", desc=True).execute()
            return response.data
        except Exception as e:
            print(f"Error getting project files: {e}")
            return []
            
    def delete_file(self, file_id: str) -> bool:
        """Deletes a file and cascades to related tables (chunks, file_keywords)."""
        try:
            # First clean up junction table manually if needed, but schema has ON DELETE CASCADE
            self.client.table("files").delete().eq("id", file_id).execute()
            return True
        except Exception as e:
            print(f"Error deleting file {file_id}: {e}")
            return False

    def upload_file_metadata(self, project_id: str, name: str, path: str, summary: str, metadata: Dict) -> Dict[str, Any]:
        """Uploads file metadata."""
        file_data = {
            "project_id": project_id,
            "name": name,
            "path": path,
            "summary": summary,
            "metadata": metadata
        }
        res = self.client.table("files").insert(file_data).execute()
        return res.data[0]

    def store_chunks(self, file_id: str, chunks: List[Dict[str, Any]]):
        """Stores chunks for a file."""
        data_to_insert = []
        for chunk in chunks:
            data_to_insert.append({
                "file_id": file_id,
                "content": chunk['content'],
                "chunk_index": chunk['chunk_index'],
                "embedding": chunk['embedding']
            })
        self.client.table("file_chunks").insert(data_to_insert).execute()

    def store_keywords(self, file_id: str, keywords: List[str]):
        """
        Stores keywords and links them to the file using RPC.
        """
        if not keywords:
            return

        try:
            # Normalize keywords
            normalized_keywords = [kw.strip().lower() for kw in keywords if kw.strip()]
            
            # Call RPC
            self.client.rpc("link_file_keywords", {
                "p_file_id": file_id,
                "p_keywords": normalized_keywords
            }).execute()
            
        except Exception as e:
            print(f"Error linking keywords for file {file_id}: {e}")

    def get_related_files(self, file_id: str) -> List[Dict[str, Any]]:
        """
        Finds related files based on shared keywords using RPC.
        Returns a list of files with a 'shared_count'.
        """
        try:
            response = self.client.rpc("get_related_files", {"p_file_id": file_id}).execute()
            return response.data
        except Exception as e:
            print(f"Error getting related files: {e}")
            return []

    def search_vectors(self, query_embedding: List[float], match_threshold: float, match_count: int, project_id: str):
        """Call the match_file_chunks RPC function."""
        params = {
            "query_embedding": query_embedding,
            "match_threshold": match_threshold,
            "match_count": match_count,
            "filter_project_id": project_id
        }
        res = self.client.rpc("match_file_chunks", params).execute()
        return res.data
