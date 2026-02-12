import sys
import os
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion import DoclingParser

def test_hierarchical_parsing():
    parser = DoclingParser()
    file_path = os.path.abspath("tests/sample_hierarchy.md")
    
    print(f"Parsing {file_path}...")
    chunks = parser.parse(file_path)
    
    print(f"Generated {len(chunks)} chunks.")
    
    # Check for hierarchy
    has_parents = False
    max_level = 0
    
    for i, chunk in enumerate(chunks):
        level = chunk.get('chunk_level', 0)
        parent = chunk.get('parent_chunk_id')
        content = chunk.get('content', '')[:50].replace('\n', ' ')
        
        print(f"[{i}] Lvl: {level} | Parent: {parent is not None} | Content: {content}...")
        print(f"    Direct Attributes: {dir(chunk)}")
        if hasattr(chunk, 'meta'):
            print(f"    Meta: {chunk.meta}")
        if hasattr(chunk, 'parent'):
            print(f"    Parent Obj: {chunk.parent}")

        
        if parent:
            has_parents = True
        if level > max_level:
            max_level = level
            
    if has_parents or max_level > 0:
        print("\n✅ Hierarchy detected!")
        print(f"Max Level: {max_level}")
    else:
        print("\n❌ No hierarchy detected (flat chunks).")

if __name__ == "__main__":
    test_hierarchical_parsing()
