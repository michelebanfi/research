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
        """
        Stores chunks for a file with metadata and reference flags.
        REQ-02: Now includes metadata JSONB column.
        REQ-04: Now includes is_reference flag.
        """
        data_to_insert = []
        for chunk in chunks:
            chunk_data = {
                "file_id": file_id,
                "content": chunk['content'],
                "chunk_index": chunk['chunk_index'],
                "embedding": chunk['embedding'],
                "metadata": chunk.get('metadata', {}),
                "is_reference": chunk.get('is_reference', False)
            }
            data_to_insert.append(chunk_data)
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

    def search_vectors(self, query_embedding: List[float], match_threshold: float, project_id: str, match_count: int = 50, include_references: bool = False):
        """Call the match_file_chunks RPC function."""
        params = {
            "query_embedding": query_embedding,
            "match_threshold": match_threshold,
            "match_count": match_count,
            "filter_project_id": project_id,
            "include_references": include_references  # Explicitly pass to avoid function overload ambiguity
        }
        res = self.client.rpc("match_file_chunks", params).execute()
        
        # REQ-03: Post-process to add metadata and file path
        # The RPC returns: id, file_id, content, similarity
        # We need to fetch: metadata (from file_chunks) and path (from files)
        results = res.data
        if not results:
            return []
            
        chunk_ids = [r['id'] for r in results]
        file_ids = list(set([r['file_id'] for r in results]))
        
        # 1. Get Chunk Metadata (page numbers, line numbers, etc.)
        chunks_meta = self.client.table("file_chunks").select("id, metadata").in_("id", chunk_ids).execute()
        meta_map = {c['id']: c['metadata'] for c in chunks_meta.data}
        
        # 2. Get File Info (path, name)
        files_info = self.client.table("files").select("id, path, name").in_("id", file_ids).execute()
        file_map = {f['id']: f for f in files_info.data}
        
        # 3. Merge
        for r in results:
            r['metadata'] = meta_map.get(r['id'], {})
            
            f_info = file_map.get(r['file_id'])
            if f_info:
                r['file_path'] = f_info['path']
                r['file_name'] = f_info['name']
                
        return results

    def store_graph_data(self, file_id: str, nodes: List[Dict[str, str]], edges: List[Dict[str, str]]):
        """
        Stores graph nodes and edges using the store_graph_data RPC.
        nodes: List of {'name': '...', 'type': '...'}
        edges: List of {'source': '...', 'target': '...', 'relation': '...'}
        """
        try:
            params = {
                "p_file_id": file_id,
                "p_nodes": nodes,
                "p_edges": edges
            }
            self.client.rpc("store_graph_data", params).execute()
        except Exception as e:
            print(f"Error storing graph data: {e}")

    def get_file_graph(self, file_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves the 1-hop neighborhood for a file.
        Returns list of edges with node properties.
        """
        try:
            response = self.client.rpc("get_file_graph", {"p_file_id": file_id}).execute()
            return response.data
        except Exception as e:
            print(f"Error getting file graph: {e}")
            return []

    def get_project_graph(self, project_id: str, limit: int = 500) -> List[Dict[str, Any]]:
        """
        Retrieves the global graph for a project.
        Returns list of edges with node properties.
        """
        try:
            params = {"p_project_id": project_id, "p_limit": limit}
            response = self.client.rpc("get_project_graph", params).execute()
            return response.data
        except Exception as e:
            print(f"Error getting project graph: {e}")
            return []

    def get_graph_traversal(self, start_node_id: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """
        Traverses graph from a start node.
        """
        try:
            params = {
                "start_node_id": start_node_id,
                "max_depth": max_depth
            }
            response = self.client.rpc("get_graph_traversal", params).execute()
            return response.data
        except Exception as e:
            print(f"Error traversing graph: {e}")
            return []

    # ==================== REQ-01: GraphRAG Methods ====================
    
    def search_nodes_by_name(self, query_terms: List[str], project_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        REQ-01: Search for nodes matching query terms.
        Used to find concepts related to user query for GraphRAG.
        """
        try:
            # Build OR conditions for ILIKE search
            # We'll search for any node name containing any of the query terms
            all_nodes = []
            for term in query_terms:
                if len(term) < 3:  # Skip very short terms
                    continue
                response = self.client.table("nodes").select(
                    "id, name, type"
                ).ilike("name", f"%{term}%").limit(limit).execute()
                all_nodes.extend(response.data)
            
            # Deduplicate by node id
            seen = set()
            unique_nodes = []
            for node in all_nodes:
                if node['id'] not in seen:
                    seen.add(node['id'])
                    unique_nodes.append(node)
            
            return unique_nodes[:limit]
        except Exception as e:
            print(f"Error searching nodes: {e}")
            return []
    
    def get_chunks_by_concepts(self, node_ids: List[str], project_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        REQ-01: Get file chunks connected to specific concept nodes.
        Used for GraphRAG to retrieve chunks via graph relationships.
        """
        try:
            if not node_ids:
                return []
            
            # Get file_ids linked to these nodes
            file_ids_response = self.client.table("files_nodes").select(
                "file_id"
            ).in_("node_id", node_ids).execute()
            
            file_ids = list(set([r['file_id'] for r in file_ids_response.data]))
            
            if not file_ids:
                return []
            
            # Get chunks from those files (filtered by project)
            chunks_response = self.client.table("file_chunks").select(
                "id, file_id, content, metadata, is_reference"
            ).in_("file_id", file_ids).eq(
                "is_reference", False  # Exclude reference chunks
            ).limit(limit).execute()
            
            # Add source info
            for chunk in chunks_response.data:
                chunk['source'] = 'graph'  # Mark as graph-retrieved
            
            # REQ-03: Augment Graph Results with File Path
            # (Metadata is already in chunks_response.data from select)
            
            chunk_file_ids = list(set([c['file_id'] for c in chunks_response.data]))
            if chunk_file_ids:
                files_info = self.client.table("files").select("id, path, name").in_("id", chunk_file_ids).execute()
                file_map = {f['id']: f for f in files_info.data}
                
                for r in chunks_response.data:
                    f_info = file_map.get(r['file_id'])
                    if f_info:
                        r['file_path'] = f_info['path']
                        r['file_name'] = f_info['name']
            
            return chunks_response.data
        except Exception as e:
            print(f"Error getting chunks by concepts: {e}")
            return []
    
    def get_related_concepts(self, node_ids: List[str], max_hops: int = 1) -> List[Dict[str, Any]]:
        """
        REQ-01: Get concepts related to the given nodes via edges.
        Expands the context for GraphRAG.
        """
        try:
            if not node_ids:
                return []
            
            related_nodes = []
            
            # Get edges where our nodes are source or target
            edges_as_source = self.client.table("edges").select(
                "target_node_id, type"
            ).in_("source_node_id", node_ids).execute()
            
            edges_as_target = self.client.table("edges").select(
                "source_node_id, type"
            ).in_("target_node_id", node_ids).execute()
            
            # Collect related node ids
            related_ids = set()
            for edge in edges_as_source.data:
                related_ids.add(edge['target_node_id'])
            for edge in edges_as_target.data:
                related_ids.add(edge['source_node_id'])
            
            # Remove original nodes
            related_ids -= set(node_ids)
            
            if related_ids:
                nodes_response = self.client.table("nodes").select(
                    "id, name, type"
                ).in_("id", list(related_ids)).execute()
                related_nodes = nodes_response.data
            
            return related_nodes
        except Exception as e:
            print(f"Error getting related concepts: {e}")
            return []

    def get_all_file_summaries(self, project_id: str) -> str:
        """
        Tool: Retrieves the high-level summaries of all files in the project.
        Useful for answering broad questions like 'What is this project about?'.
        """
        try:
            # Fetch only name and summary columns
            response = self.client.table("files").select("name, summary").eq("project_id", project_id).execute()
            
            if not response.data:
                return "No files found in this project."
                
            # Format them into a single context string
            context_parts = []
            for f in response.data:
                name = f.get('name', 'Unknown File')
                summary = f.get('summary', 'No summary available.')
                context_parts.append(f"File: {name}\nSummary: {summary}")
                
            return "\n\n".join(context_parts)
        except Exception as e:
            print(f"Error getting file summaries: {e}")
            return ""

    # ==================== REQ-DATA-02: Semantic Auto-Clustering ====================
    
    def run_semantic_clustering(self, project_id: str, min_cluster_size: int = 3) -> Dict[str, Any]:
        """
        REQ-DATA-02: Identifies dense node clusters and creates Topic nodes.
        
        Algorithm:
        1. Fetch all edges for the project
        2. Build adjacency counts per node
        3. Identify nodes with high connectivity (potential cluster centers)
        4. Group connected concepts into clusters
        5. Create "Topic" nodes linking each cluster
        
        Args:
            project_id: The project to cluster
            min_cluster_size: Minimum nodes to form a cluster
            
        Returns:
            Dict with 'topics_created' count and 'clusters' list
        """
        try:
            # 1. Get all edges for the project
            edges_data = self.get_project_graph(project_id, limit=1000)
            
            if not edges_data:
                return {"topics_created": 0, "clusters": []}
            
            # 2. Build adjacency maps
            from collections import defaultdict
            adjacency = defaultdict(set)  # node_name -> set of connected node names
            node_types = {}  # node_name -> type
            
            for edge in edges_data:
                source = edge.get('source_name', '')
                target = edge.get('target_name', '')
                source_type = edge.get('source_type', 'Concept')
                target_type = edge.get('target_type', 'Concept')
                
                if source and target:
                    adjacency[source].add(target)
                    adjacency[target].add(source)
                    node_types[source] = source_type
                    node_types[target] = target_type
            
            # 3. Find cluster centers (nodes with high connectivity)
            cluster_centers = []
            for node, neighbors in adjacency.items():
                # Skip nodes that are already Topic type
                if node_types.get(node) == 'Topic':
                    continue
                if len(neighbors) >= min_cluster_size:
                    cluster_centers.append((node, len(neighbors), neighbors))
            
            # Sort by connectivity
            cluster_centers.sort(key=lambda x: x[1], reverse=True)
            
            # 4. Create clusters (greedy approach - avoid overlapping)
            used_nodes = set()
            clusters = []
            
            for center, count, neighbors in cluster_centers:
                if center in used_nodes:
                    continue
                    
                # Get unused neighbors for this cluster
                cluster_members = [n for n in neighbors if n not in used_nodes]
                
                if len(cluster_members) >= min_cluster_size - 1:  # -1 because center is also a member
                    cluster = {
                        "center": center,
                        "members": [center] + cluster_members[:min_cluster_size * 2],  # Cap members
                        "size": len(cluster_members) + 1
                    }
                    clusters.append(cluster)
                    
                    # Mark nodes as used
                    used_nodes.add(center)
                    for n in cluster_members[:min_cluster_size * 2]:
                        used_nodes.add(n)
            
            # 5. Create Topic nodes for each cluster
            topics_created = 0
            for i, cluster in enumerate(clusters):
                # Generate topic name from cluster members
                topic_name = f"Topic: {cluster['center']}"
                
                # Create Topic node
                topic_node = {"name": topic_name, "type": "Topic"}
                
                # Create edges from Topic to all cluster members
                topic_edges = []
                for member in cluster["members"]:
                    topic_edges.append({
                        "source": topic_name,
                        "target": member,
                        "relation": "groups",
                        "source_type": "Topic",
                        "target_type": node_types.get(member, "Concept")
                    })
                
                # Store the topic node and edges (using first file in project as anchor)
                files = self.get_project_files(project_id)
                if files:
                    file_id = files[0]['id']
                    self.store_graph_data(file_id, [topic_node], topic_edges)
                    topics_created += 1
            
            return {
                "topics_created": topics_created,
                "clusters": [
                    {"name": c["center"], "size": c["size"], "members": c["members"][:5]}  # Limit for output
                    for c in clusters
                ]
            }
            
        except Exception as e:
            print(f"Error in semantic clustering: {e}")
            return {"topics_created": 0, "clusters": [], "error": str(e)}
