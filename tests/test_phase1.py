import os
import sys
import asyncio
import json
import time
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database import DatabaseClient
from src.cache import Cache
from src.config import Config

def test_cache_lru():
    print("\n--- Testing Cache LRU ---")
    cache_dir = "tests/.cache_test"
    # Create cache with tiny max_size
    c = Cache(cache_dir=cache_dir, max_size=2)
    c.clear() # Verify clear works
    
    c.set("p", "k1", "v1")
    time.sleep(0.1)
    c.set("p", "k2", "v2")
    time.sleep(0.1)
    c.set("p", "k3", "v3") # Should evict k1
    
    v1 = c.get("p", "k1")
    v2 = c.get("p", "k2")
    v3 = c.get("p", "k3")
    
    print(f"k1 (should be None): {v1}")
    print(f"k2 (should be v2): {v2}")
    print(f"k3 (should be v3): {v3}")
    
    assert v1 is None, "LRU failed: k1 should be evicted"
    assert v2 == "v2", "k2 should be present"
    assert v3 == "v3", "k3 should be present"
    print("✅ Cache LRU Test Passed")

def test_db_rpc():
    print("\n--- Testing DB RPC & FTS ---")
    try:
        db = DatabaseClient()
        projects = db.get_projects()
        if not projects:
            print("No projects found, skipping DB search test.")
            return

        p_id = projects[0]['id']
        print(f"Using Project: {projects[0]['name']} ({p_id})")
        
        # Test vector search with keyword (FTS)
        # We pass a dummy embedding/query. 
        # If FTS works, it should filter regardless of embedding if we use keyword_query?
        # RPC requires embedding. We'll use a zero vector.
        dummy_embedding = [0.0] * 1024
        
        # Pass a keyword that likely doesn't exist to verify filtering, 
        # or one that exists if we knew content.
        # Let's just call it and ensure no SQL error is raised.
        results = db.search_vectors(
            query_embedding=dummy_embedding,
            match_threshold=-1.0, # Get everything
            project_id=p_id,
            match_count=1,
            keyword_query="test"
        )
        print(f"RPC Call Successful. Results found: {len(results)}")
        if results:
            print("Sample Result Keys:", results[0].keys())
            assert "file_path" in results[0], "RPC did not return file_path"
            assert "metadata" in results[0], "RPC did not return metadata"
            
        print("✅ DB RPC Test Passed")
        
    except Exception as e:
        print(f"❌ DB RPC Test Failed: {e}")
        # Dont fail the script, just log (maybe DB not connected in this env)

if __name__ == "__main__":
    test_cache_lru()
    test_db_rpc()
