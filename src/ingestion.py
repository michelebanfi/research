import os
import ast
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any
from docling.document_converter import DocumentConverter
from docling.chunking import HierarchicalChunker

class FileType(Enum):
    CODE = "code"
    DOCUMENT = "document"
    UNKNOWN = "unknown"

class FileRouter:
    @staticmethod
    def route(file_path: str) -> FileType:
        ext = Path(file_path).suffix.lower()
        if ext == ".py":
            return FileType.CODE
        elif ext in [".pdf", ".md", ".docx", ".txt"]:
            return FileType.DOCUMENT
        return FileType.UNKNOWN

class DoclingParser:
    def __init__(self):
        self.converter = DocumentConverter()
        self.chunker = HierarchicalChunker()

    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parses a document using Docling and HierarchicalChunker.
        Returns a list of chunks (dicts) with content and metadata.
        """
        try:
            result = self.converter.convert(file_path)
            # Use Docling's hierarchical chunker
            chunk_iter = self.chunker.chunk(result.document)
            
            chunks = []
            for i, chunk in enumerate(chunk_iter):
                # chunk.text contains the content
                # chunk.meta contains metadata like headers
                chunks.append({
                    "content": chunk.text,
                    "chunk_index": i,
                    "metadata": chunk.meta.export_json_dict() if hasattr(chunk.meta, 'export_json_dict') else str(chunk.meta)
                })
                
            return chunks
        except Exception as e:
            print(f"Error parsing document {file_path}: {e}")
            return []

class ASTParser:
    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parses Python code using AST to split by classes and methods granularly.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
            
            tree = ast.parse(source)
            chunks = []
            chunk_index = 0
            
            lines = source.splitlines(keepends=True)
            
            def get_segment(node):
                # lineno is 1-indexed, end_lineno is inclusive
                if not hasattr(node, 'lineno') or not hasattr(node, 'end_lineno'):
                    return ""
                return "".join(lines[node.lineno-1 : node.end_lineno])

            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    # For a class, we want to extract its docstring/signature as one chunk
                    # And then each method as a separate chunk
                    
                    # 1. Class Header / Docstring
                    # We can try to approximate by taking everything up to the first method or just the logic
                    # A simple way involves getting the class source and stripping methods, or just taking the top.
                    
                    class_name = node.name
                    # Let's create a chunk for the class definition itself (excluding body methods if possible, 
                    # but AST doesn't give clean "header only" range easily without traversing children).
                    # Simplified: Chunk the whole class docstring + signature if present?
                    # Or just: 
                    # For each method in body -> chunk with metadata parent_class = class_name
                    # For the class itself -> maybe just a summary?
                    
                    # Iterate methods
                    methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                    
                    # If we want to capture class-level vars or docstring, we should
                    # But the requirement ING-04 says: "extract methods as individual chunks... while preserving the parent class name"
                    
                    for method in methods:
                        method_content = get_segment(method)
                        chunks.append({
                            "content": method_content,
                            "chunk_index": chunk_index,
                            "type": "method",
                            "name": method.name,
                            "parent_class": class_name,
                            "metadata": {"parent_class": class_name}
                        })
                        chunk_index += 1
                        
                    # What about the class itself (e.g. valid Pydantic models with no methods)?
                    # If it has no methods, we should chunk the whole class.
                    if not methods:
                        content = get_segment(node)
                        chunks.append({
                            "content": content,
                            "chunk_index": chunk_index,
                            "type": "class",
                            "name": class_name,
                            "metadata": {"is_class_def": True}
                        })
                        chunk_index += 1
                        
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Standalone function
                    content = get_segment(node)
                    chunks.append({
                        "content": content,
                        "chunk_index": chunk_index,
                        "type": "function",
                        "name": node.name
                    })
                    chunk_index += 1
                else:
                    # Imports, constants, globals
                    # Optionally group them?
                    pass
                    
            return chunks

        except Exception as e:
            print(f"Error parsing code {file_path}: {e}")
            return []
