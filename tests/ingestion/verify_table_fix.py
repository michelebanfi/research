
import sys
import logging
from src.ingestion import DoclingParser

# Configure logging
logging.basicConfig(level=logging.INFO)

def verify_fix(file_path):
    print(f"Verifying VLM table fix for: {file_path}")
    
    try:
        # Initialize parser
        parser = DoclingParser(enable_vlm=True)
        
        # Parse
        chunks = parser.parse(file_path)
        
        table_chunks = [c for c in chunks if c.get('metadata', {}).get('is_table')]
        print(f"Table chunks found: {len(table_chunks)}")
        
        for i, chunk in enumerate(table_chunks):
            print(f"\n--- Table Chunk {i+1} ---")
            content = chunk.get('content', '')
            print(f"Content start: {content[:100]}")
            
            # Verification logic
            if "|" in content and "-|-" in content: # Simple markdown table check
                print("✅ SUCCESS: Content looks like a Markdown table.")
            else:
                print("❌ FAILURE: Content does NOT look like a Markdown table.")
                print(f"Full content:\n{content}")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    file_path = "static/uploads/High-threshold and low-overhead fault-tolerant quantum memory.pdf"
    verify_fix(file_path)
