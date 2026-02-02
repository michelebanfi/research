
import os
import ast
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any
from docling.document_converter import DocumentConverter

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
        elif ext in [".pdf", ".md", ".docx", ".txt"]: # Docling supports these and more
            return FileType.DOCUMENT
        return FileType.UNKNOWN

class DoclingParser:
    def __init__(self):
        self.converter = DocumentConverter()

    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parses a document using Docling.
        Returns a list of chunks (dicts) with content and metadata (like chunk index).
        For now, we'll extract the full markdown/text and chunk it simply or rely on Docling's structure if available.
        To keep it simple as per requirements: "converted to Markdown, and chunked respecting the layout hierarchy."
        """
        try:
            result = self.converter.convert(file_path)
            # result.document is the Docling document object.
            # We can export to markdown.
            md_content = result.document.export_to_markdown()
            
            # Simple chunking by double newline for now, or we could walk the document structure.
            # Requirement says "chunked respecting the layout hierarchy".
            # Docling's export_to_markdown preserves layout. Splitting by headers (#) might be better.
            # Let's do a naive split by 1000 chars for now or headers if possible. 
            # Better: split by double newlines and group.
            
            chunks = []
            raw_chunks = md_content.split("\n\n")
            current_chunk = ""
            chunk_index = 0
            
            for raw in raw_chunks:
                if len(current_chunk) + len(raw) < 1000:
                    current_chunk += raw + "\n\n"
                else:
                    if current_chunk.strip():
                        chunks.append({
                            "content": current_chunk.strip(),
                            "chunk_index": chunk_index
                        })
                        chunk_index += 1
                    current_chunk = raw + "\n\n"
            
            if current_chunk.strip():
                chunks.append({
                    "content": current_chunk.strip(),
                    "chunk_index": chunk_index
                })
                
            return chunks
        except Exception as e:
            print(f"Error parsing document {file_path}: {e}")
            return []

class ASTParser:
    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parses Python code using AST to split by classes and functions.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
            
            tree = ast.parse(source)
            chunks = []
            chunk_index = 0
            
            # Helper to extract source segment
            lines = source.splitlines(keepends=True)
            
            def get_segment(node):
                # lineno is 1-indexed, end_lineno is inclusive
                return "".join(lines[node.lineno-1 : node.end_lineno])

            # Walk the tree for top-level classes and functions
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    content = get_segment(node)
                    chunks.append({
                        "content": content,
                        "chunk_index": chunk_index,
                        "type": type(node).__name__,
                        "name": node.name
                    })
                    chunk_index += 1
                else:
                    # For other top-level statements (imports, assignments), we could group them.
                    # Ignored for now to keep it semantic as requested (Classes/Functions).
                    pass
                    
            return chunks

        except Exception as e:
            print(f"Error parsing code {file_path}: {e}")
            return []
