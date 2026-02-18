
import sys
import logging
from src.ingestion import DoclingParser
from docling.document_converter import DocumentConverter

# Configure logging
logging.basicConfig(level=logging.INFO)

def debug_table_export(file_path):
    print(f"Debugging table export for: {file_path}")
    
    try:
        # Initialize parser
        parser = DoclingParser(enable_vlm=True)
        converter = parser.converter
        # Convert
        print("Converting document...")
        result = converter.convert(file_path)
        doc = result.document

        # Inspect chunks
        print("\n--- Inspecting Chunks ---")
        chunker = parser.chunker
        chunks = chunker.chunk(doc)
        
        for i, chunk in enumerate(chunks):
            is_table = parser._is_table_chunk(chunk)
            if is_table:
                print(f"\n[Chunk {i}] Identified as Table")
                # print(f"Content: {chunk.text[:100]}...")
                
                if hasattr(chunk, 'meta'):
                    print(f"Meta: {chunk.meta}")
                    if hasattr(chunk.meta, 'doc_items'):
                        print(f"Doc Items: {chunk.meta.doc_items}")
                        for item in chunk.meta.doc_items:
                            print(f"  Item Label: {item.label}")
                            if hasattr(item, 'self_ref'):
                                print(f"  Item Ref: {item.self_ref}")
                            elif hasattr(item, 'ref'):
                                print(f"  Item Ref (alt): {item.ref}")
                            else:
                                print(f"  Item Dict: {item.__dict__}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    file_path = "static/uploads/High-threshold and low-overhead fault-tolerant quantum memory.pdf"
    debug_table_export(file_path)
