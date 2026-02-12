import sys
import os
import asyncio
import nest_asyncio

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database import DatabaseClient
from src.ai_engine import AIEngine
from src.ingestion import DoclingParser

nest_asyncio.apply()

async def run_verification():
    print("=== Starting Phase 3 Verification ===")
    
    # 1. Setup
    db = DatabaseClient()
    ai = AIEngine()
    parser = DoclingParser()
    
    project_name = "Phase3_Verify_Hierarchy"
    print(f"Creating/Finding project: {project_name}")
    project = db.create_project(project_name)
    if not project:
        # Try finding it
        projects = db.get_projects()
        project = next((p for p in projects if p['name'] == project_name), None)
        if not project:
            print("Failed to create/find project")
            return
            
    project_id = project['id']
    print(f"Project ID: {project_id}")
    
    # Clean up previous runs
    files = db.get_project_files(project_id)
    for f in files:
        db.delete_file(f['id'])
    
    # 2. Ingest Sample File (Hierarchical)
    sample_file = os.path.join(os.path.dirname(__file__), "sample_hierarchy.md")
    # Ensure sample file exists with good structure
    with open(sample_file, "w") as f:
        f.write("# Main Section\n\nIntro text.\n\n## Subsection A\nContent of A.\n\n## Subsection B\nContent of B which is very specific.")
            
    print(f"Ingesting {sample_file}...")
    
    try:
        # Parse
        chunks = parser.parse(sample_file)
        print(f"Parsed {len(chunks)} chunks.")
        
        # Simulate Embeddings & Storage
        print("Generating embeddings...")
        for chunk in chunks:
            # Generate embedding
            text = chunk.get('embedding_text', chunk['content'])
            chunk['embedding'] = await ai.generate_embedding_async(text)
            
        # Store File Metadata
        print("Storing metadata...")
        file_meta = db.upload_file_metadata(
            project_id=project_id,
            name="sample_hierarchy.md",
            path=str(sample_file),
            summary="A test file for hierarchy.",
            metadata={"keywords": ["test"]}
        )
        
        if file_meta:
            file_id = file_meta['id']
            db.store_chunks(file_id, chunks)
            print("Storage complete.")
        else:
            print("Failed to upload file metadata.")
            return

    except Exception as e:
        print(f"Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Verify Hierarchy in DB
    print("\nVerifying Database Chunks...")
    files = db.get_project_files(project_id)
    if not files:
        print("No files found in DB.")
        return
        
    file_id = files[0]['id']
    chunks_resp = db.client.table("file_chunks").select("*").eq("file_id", file_id).execute()
    db_chunks = chunks_resp.data
    
    # Identify parents (synthetic sections)
    parent_ids = set()
    for c in db_chunks:
        if c.get('parent_chunk_id'):
            parent_ids.add(c['parent_chunk_id'])
            
    parents = [c for c in db_chunks if c['id'] in parent_ids]
    leaves = [c for c in db_chunks if c['id'] not in parent_ids]
    
    print(f"Total Chunks in DB: {len(db_chunks)}")
    print(f"Identified Parents (referenced by others): {len(parents)}")
    
    print("\n--- Parent Chunks ---")
    for p in parents:
        print(f"ID: {p['id'][:8]}... | Level: {p.get('chunk_level')} | Content: {p['content'][:40]}...")

    print("\n--- Leaf Chunks ---")
    hierarchy_found = False
    for l in leaves:
        pid = l.get('parent_chunk_id')
        print(f"Leaf: {l['content'][:40]}... | ParentID: {pid}")
        if pid:
            hierarchy_found = True
            
    if hierarchy_found:
        print("\nSUCCESS: Hierarchy detected (Leaf points to Parent).")
    else:
        print("\nFAILURE: No hierarchy detected (No parent_chunk_ids found).")
        
    # 4. Precise Retrieval Test
    query = "Content of B"
    print(f"\nTesting Advanced Retrieval for: '{query}'")
    
    # Use retrieval with fusion and reranking
    results = await ai.retrieve_advanced(query, db, project_id, limit=3)
    
    for i, res in enumerate(results):
        print(f"Result {i+1}: {res['content'][:60]}... (RRF Score: {res.get('rrf_score')})")
        
    if results and "Content of B" in results[0]['content']:
        print("SUCCESS: Retrieval found target.")
    else:
        print("FAILURE: Retrieval missed target.")

if __name__ == "__main__":
    asyncio.run(run_verification())
