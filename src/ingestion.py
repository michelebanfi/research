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
                    class_name = node.name
                    
                    # 1. Capture Class Header / Docstring Context
                    # We'll take the first few lines of the class (docstring/init) to give context
                    # Or simpler: just "Class: {class_name}" plus docstring if available.
                    
                    class_doc = ast.get_docstring(node)
                    class_context = f"Parent Class: {class_name}\n"
                    if class_doc:
                        class_context += f"Class Docstring: {class_doc}\n"
                    
                    # Iterate methods
                    methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                    
                    for method in methods:
                        method_content = get_segment(method)
                        # Prepend context
                        full_content = f"{class_context}\n{method_content}"
                        
                        chunks.append({
                            "content": full_content,
                            "chunk_index": chunk_index,
                            "type": "method",
                            "name": method.name,
                            "parent_class": class_name,
                            "metadata": {"parent_class": class_name, "context_added": True}
                        })
                        chunk_index += 1
                        
                    # If it has no methods, chunk the whole class
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
