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
    """
    Parses documents using Docling with enhanced metadata extraction.
    Implements REQ-02 (granular metadata), REQ-04 (bibliography filtering), REQ-07 (table handling).
    """
    
    # REQ-04: Patterns to identify reference/bibliography sections
    REFERENCE_SECTION_PATTERNS = [
        "references", "bibliography", "works cited", "citations",
        "literature cited", "sources", "endnotes", "footnotes"
    ]
    
    def __init__(self):
        self.converter = DocumentConverter()
        self.chunker = HierarchicalChunker()
    
    def _is_reference_section(self, text: str, metadata: Dict[str, Any]) -> bool:
        """
        REQ-04: Detect if a chunk is part of a references/bibliography section.
        Uses both header detection and content heuristics.
        """
        text_lower = text.lower()
        
        # Check headers in metadata
        headings = metadata.get("headings", [])
        for heading in headings:
            heading_lower = heading.lower() if isinstance(heading, str) else ""
            for pattern in self.REFERENCE_SECTION_PATTERNS:
                if pattern in heading_lower:
                    return True
        
        # Content heuristics: typical citation patterns like "[1]", "(Author, 2020)"
        import re
        # More than 5 citation-like patterns in a short chunk suggests it's a reference list
        citation_pattern = r'\[\d+\]|\(\w+,?\s*\d{4}\)|\d+\.\s+[A-Z][a-z]+,\s+[A-Z]\.'
        matches = re.findall(citation_pattern, text)
        if len(matches) > 5 and len(text) < 2000:
            return True
        
        return False
    
    def _is_table_chunk(self, chunk) -> bool:
        """
        REQ-07: Detect if a chunk contains table data.
        Docling often marks tables in metadata or uses structured formats.
        """
        # Check if chunk metadata indicates table
        if hasattr(chunk, 'meta'):
            meta_dict = chunk.meta.export_json_dict() if hasattr(chunk.meta, 'export_json_dict') else {}
            if meta_dict.get("doc_item_type") == "table" or meta_dict.get("is_table"):
                return True
        
        # Heuristic: Markdown table patterns (|---|) or tab-separated rows
        text = chunk.text if hasattr(chunk, 'text') else ""
        if "|---" in text or text.count("\t") > 5:
            return True
        
        return False
    
    def _extract_granular_metadata(self, chunk, chunk_index: int) -> Dict[str, Any]:
        """
        REQ-02: Extract granular metadata from Docling chunk.
        Includes page numbers, section headers, bounding boxes if available.
        """
        metadata = {}
        
        if hasattr(chunk, 'meta'):
            if hasattr(chunk.meta, 'export_json_dict'):
                raw_meta = chunk.meta.export_json_dict()
            else:
                raw_meta = {"raw": str(chunk.meta)}
            
            # Extract specific fields for structured access
            metadata["headings"] = raw_meta.get("headings", [])
            metadata["doc_item_type"] = raw_meta.get("doc_item_type", "text")
            
            # Page info (Docling provides this for PDFs)
            if "page_no" in raw_meta:
                metadata["page_number"] = raw_meta["page_no"]
            elif "page" in raw_meta:
                metadata["page_number"] = raw_meta["page"]
            
            # Bounding box if available
            if "bbox" in raw_meta:
                metadata["bounding_box"] = raw_meta["bbox"]
            
            # Section hierarchy
            if metadata["headings"]:
                metadata["section_header"] = metadata["headings"][-1] if metadata["headings"] else None
        
        metadata["chunk_index"] = chunk_index
        return metadata

    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parses a document using Docling and HierarchicalChunker.
        Returns a list of chunks with enhanced metadata, reference flags, and table handling.
        """
        try:
            result = self.converter.convert(file_path)
            chunk_iter = self.chunker.chunk(result.document)
            
            chunks = []
            for i, chunk in enumerate(chunk_iter):
                # REQ-02: Extract granular metadata
                metadata = self._extract_granular_metadata(chunk, i)
                
                # REQ-04: Check if this is a reference section
                is_reference = self._is_reference_section(chunk.text, metadata)
                
                # REQ-07: Handle tables semantically
                is_table = self._is_table_chunk(chunk)
                
                chunk_data = {
                    "content": chunk.text,
                    "chunk_index": i,
                    "metadata": metadata,
                    "is_reference": is_reference,
                    "is_table": is_table,
                }
                
                # REQ-07: For tables, we'll also store a summary hint for embedding
                if is_table:
                    # The AI engine will generate a summary for embedding
                    # but we preserve raw content for display
                    chunk_data["metadata"]["is_table"] = True
                    chunk_data["metadata"]["table_raw_content"] = chunk.text
                    chunk_data["embedding_text"] = f"[TABLE] {chunk.text[:500]}"  # Truncated for embedding
                
                chunks.append(chunk_data)
                
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
