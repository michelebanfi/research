
import sys
import logging
from src.ingestion import DoclingParser
import json

# Configure logging
logging.basicConfig(level=logging.INFO)

def verify_fix(file_path):
    print(f"Verifying ingestion fix for: {file_path}")
    
    try:
        # Initialize parser
        parser = DoclingParser(enable_vlm=True)
        
        # Parse document
        print("Parsing document...")
        chunks = parser.parse(file_path)
        
        print(f"Generated {len(chunks)} chunks")
        
        table_chunks = [c for c in chunks if c.get("metadata", {}).get("is_table")]
        print(f"Found {len(table_chunks)} table chunks")
        
        for i, chunk in enumerate(table_chunks):
            print(f"\n--- Table Chunk {i} ---")
            content = chunk.get("content", "")
            print(f"Content Start: {content[:100]}...")
            
            # check if it looks like markdown
            if "|---" in content or "| ---" in content:
                print("✅ Content appears to be Markdown table")
            else:
                print("❌ Content does NOT look like Markdown table")
                print(f"Full content: {content}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    file_path = "static/uploads/High-threshold and low-overhead fault-tolerant quantum memory.pdf"
    verify_fix(file_path)
