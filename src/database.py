
import os
from supabase import create_client, Client
from typing import List, Dict, Any, Optional

class DatabaseClient:
    def __init__(self):
        self.url = os.environ.get("SUPABASE_URL")
        self.key = os.environ.get("SUPABASE_KEY")
        if not self.url or not self.key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
        self.client: Client = create_client(self.url, self.key)

    def create_project(self, name: str) -> Dict[str, Any]:
        """Creates a new project and returns the inserted row."""
        try:
            data, count = self.client.table("projects").insert({"name": name}).execute()
            if data and len(data) > 0:
                 return data[1][0]
            return None
        except Exception as e:
            print(f"Error creating project: {e}")
            if hasattr(data, 'data'):
                return data.data[0]
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
        Stores keywords and links them to the file.
        1. Ensure each keyword exists in `keywords` table (ID lookup).
        2. Create entry in `file_keywords`.
        """
        if not keywords:
            return

        keyword_ids = []
        for kw in keywords:
            kw = kw.lower().strip()
            # Try to select first
            try:
                res = self.client.table("keywords").select("id").eq("keyword", kw).execute()
                if res.data:
                    keyword_ids.append(res.data[0]['id'])
                else:
                    # Insert
                    res_ins = self.client.table("keywords").insert({"keyword": kw}).execute()
                    if res_ins.data:
                         keyword_ids.append(res_ins.data[0]['id'])
            except Exception as e:
                # Handle race condition or unique constraint error if parallel
                print(f"Error storing keyword {kw}: {e}")
                # Retry select
                res = self.client.table("keywords").select("id").eq("keyword", kw).execute()
                if res.data:
                    keyword_ids.append(res.data[0]['id'])
        
        # Link to file
        links = [{"file_id": file_id, "keyword_id": kid} for kid in keyword_ids]
        if links:
            try:
                self.client.table("file_keywords").insert(links).execute()
            except Exception as e:
                 print(f"Error linking keywords for file {file_id}: {e}")

    def get_related_files(self, file_id: str) -> List[Dict[str, Any]]:
        """
        Finds related files based on shared keywords.
        Returns a list of files with a 'shared_count' and 'keywords'.
        """
        # 1. Get keywords for this file
        try:
            # We need to perform a join or subquery. Supabase-py is limited, but we can use RPC or raw SQL or multiple queries.
            # Let's use RPC if complex, or multiple queries for simplicity here since no complex graph RPC exists yet.
            
            # Get keyword IDs for the file
            fk_res = self.client.table("file_keywords").select("keyword_id").eq("file_id", file_id).execute()
            kw_ids = [item['keyword_id'] for item in fk_res.data]
            
            if not kw_ids:
                return []
            
            # Find other files with these keywords
            # Using 'in' filter
            # We want to group by file_id and count matches.
            # This is hard with simple client calls.
            # Let's fetch all file_keywords where keyword_id in kw_ids
            
            # Assuming kw_ids list is not huge
            all_matches = []
            # Splitting in chunks if too many keywords, for now assuming < 20
            target_ids_tuple = tuple(kw_ids)
            # Supabase client .in_ takes a list
            res_matches = self.client.table("file_keywords").select("file_id, keyword_id").in_("keyword_id", kw_ids).execute()
            
            # Process in python
            file_counts = {}
            for item in res_matches.data:
                fid = item['file_id']
                if fid == file_id:
                    continue
                file_counts[fid] = file_counts.get(fid, 0) + 1
            
            # Sort by count
            sorted_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Get file details for top 5
            top_files = []
            for fid, count in sorted_files[:5]:
                f_res = self.client.table("files").select("*").eq("id", fid).execute()
                if f_res.data:
                    file_info = f_res.data[0]
                    file_info['shared_count'] = count
                    top_files.append(file_info)
            
            return top_files

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
