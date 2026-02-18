
import os
from src.database import DatabaseClient

def inspect_chunks():
    db = DatabaseClient()
    files = db.client.table("files").select("id, name").eq("name", "High-threshold and low-overhead fault-tolerant quantum memory.pdf").execute()
    
    if not files.data:
        print("File not found in DB")
        return

    file_id = files.data[0]['id']
    print(f"File ID: {file_id}")
    
    chunks = db.get_file_chunks(file_id)
    print(f"Total chunks: {len(chunks)}")
    
    table_chunks = [c for c in chunks if c.get('metadata', {}).get('is_table')]
    print(f"Table chunks in DB: {len(table_chunks)}")
    
    for i, chunk in enumerate(table_chunks):
        print(f"Chunk {i} ID: {chunk['id']}")
        print(f"Content Preview: {chunk['content'][:100]}")
        print(f"Metadata: {chunk['metadata']}")

if __name__ == "__main__":
    inspect_chunks()
