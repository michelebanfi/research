import os
import uuid
import ast
import asyncio
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HierarchicalChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel.vlm_model_specs import GRANITEDOCLING_OLLAMA
from docling.pipeline.vlm_pipeline import VlmPipeline

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
    REQ-NLP-01: Also detects Abstract/Conclusion sections for key claim extraction.
    REQ-FIX-C: Semantic Table Embedding - stores both content and embedding_content for tables.
    """
    
    # REQ-04: Patterns to identify reference/bibliography sections
    REFERENCE_SECTION_PATTERNS = [
        "references", "bibliography", "works cited", "citations",
        "literature cited", "sources", "endnotes", "footnotes"
    ]
    
    # REQ-NLP-01: Patterns to identify sections for key claim extraction
    KEY_CLAIM_SECTION_PATTERNS = {
        "abstract": ["abstract", "summary", "overview"],
        "conclusion": ["conclusion", "conclusions", "summary", "discussion", "findings"],
        "results": ["results", "contributions", "main findings"]
    }
    
    def __init__(self, ai_engine=None, enable_vlm: bool = True):
        self.ai_engine = ai_engine
        self.chunker = HierarchicalChunker()
        
        # Configure Docling Converter
        if enable_vlm:
            try:
                # Configure VLM Pipeline Options for Ollama
                pipeline_options = VlmPipelineOptions()
                pipeline_options.vlm_options = GRANITEDOCLING_OLLAMA
                # Ensure correct model tag is used
                pipeline_options.vlm_options.params["model"] = "ibm/granite-docling:latest"
                
                # Enable remote services for Ollama API
                try:
                    pipeline_options.enable_remote_services = True
                except:
                    pass

                self.converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(
                            pipeline_cls=VlmPipeline,
                            pipeline_options=pipeline_options
                        )
                    }
                )
                self.converter.allowed_remote_services = True # Critical for Ollama
                print("DoclingParser initialized with Ollama VLM pipeline.")
            except Exception as e:
                print(f"Failed to initialize VLM pipeline: {e}. Falling back to standard parser.")
                self.converter = DocumentConverter()
        else:
            self.converter = DocumentConverter()
    
    def _detect_key_claim_section(self, metadata: Dict[str, Any]) -> str:
        """
        REQ-NLP-01: Detect if a chunk is from a section relevant for key claim extraction.
        
        Returns:
            Section type ("abstract", "conclusion", "results") or empty string if not relevant.
        """
        headings = metadata.get("headings", [])
        for heading in headings:
            heading_lower = heading.lower() if isinstance(heading, str) else ""
            for section_type, patterns in self.KEY_CLAIM_SECTION_PATTERNS.items():
                for pattern in patterns:
                    if pattern in heading_lower:
                        return section_type
        return ""
    
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
            
            # Check for doc_items list (Docling v2 structure)
            if "doc_items" in meta_dict:
                for item in meta_dict["doc_items"]:
                    if item.get("label") == "table":
                        return True
            
            # Legacy/Fallback checks
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
            
            # Determine doc_item_type from doc_items if available
            doc_item_type = raw_meta.get("doc_item_type", "text")
            if "doc_items" in raw_meta and raw_meta["doc_items"]:
                # Use the label of the first item as the type
                doc_item_type = raw_meta["doc_items"][0].get("label", "text")
            
            metadata["doc_item_type"] = doc_item_type
            
            # Page info (Docling provides this for PDFs)
            # Check doc_items for provenance first
            page_no = None
            bbox = None
            
            if "doc_items" in raw_meta:
                for item in raw_meta["doc_items"]:
                    if "prov" in item and item["prov"]:
                        prov = item["prov"][0]
                        page_no = prov.get("page_no")
                        bbox = prov.get("bbox")
                        break
            
            if page_no:
                metadata["page_number"] = page_no
            elif "page_no" in raw_meta:
                metadata["page_number"] = raw_meta["page_no"]
            elif "page" in raw_meta:
                metadata["page_number"] = raw_meta["page"]
            
            # Bounding box if available
            if bbox:
                metadata["bounding_box"] = bbox
            elif "bbox" in raw_meta:
                metadata["bounding_box"] = raw_meta["bbox"]
            
            # Section hierarchy
            if metadata["headings"]:
                metadata["section_header"] = metadata["headings"][-1] if metadata["headings"] else None
        
        metadata["chunk_index"] = chunk_index
        return metadata


    async def _generate_table_summary_async(self, table_content: str) -> str:
        """
        REQ-FIX-C: Generate a 1-sentence summary of what the table shows.
        Used for semantic embedding of table content.
        """
        if not self.ai_engine:
            return table_content[:500]
        
        try:
            prompt = f"""Generate a very brief (1 sentence, max 100 characters) description of what this table shows.
Focus on the main data points or comparison being made.

Table:
{table_content[:1500]}

Description:"""
            
            summary = await self.ai_engine._openrouter_generate(prompt)
            return summary.strip() if summary else table_content[:500]
        except Exception as e:
            print(f"Error generating table summary: {e}")
            return table_content[:500]

    async def process_table_summaries_async(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        REQ-FIX-C: Process tables to generate semantic summaries for embedding.
        This should be called after parsing to populate embedding_content for table chunks.
        """
        if not self.ai_engine:
            return chunks
        
        table_chunks = [c for c in chunks if c.get('is_table', False)]
        if not table_chunks:
            return chunks
        
        tasks = []
        chunk_map = {}
        
        for chunk in table_chunks:
            task = self._generate_table_summary_async(chunk.get('content', ''))
            tasks.append(task)
            chunk_map[id(task)] = chunk
        
        if not tasks:
            return chunks
        
        summaries = await asyncio.gather(*tasks)
        
        for chunk, summary in zip(table_chunks, summaries):
            chunk['embedding_content'] = summary
            chunk['embedding_text'] = summary
        
        return chunks

    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parses a document using Docling and HierarchicalChunker.
        Returns a list of chunks with enhanced metadata, reference flags, and table handling.
        Implements Synthetic Hierarchical Chunking (Small-to-Big) by aggregating leaves into parent sections.
        """
        try:
            result = self.converter.convert(file_path)
            chunk_iter = self.chunker.chunk(result.document)
            
            leaf_chunks = []
            
            # Map of Heading Path (tuple) -> Parent Chunk Data
            # This allows us to aggregate content for sections
            section_map = {} 
            
            docling_chunks = list(chunk_iter)
            
            for i, chunk in enumerate(docling_chunks):
                chunk_uuid = str(uuid.uuid4())
                
                # REQ-02: Extract granular metadata
                metadata = self._extract_granular_metadata(chunk, i)
                
                # REQ-04: Check if this is a reference section
                is_reference = self._is_reference_section(chunk.text, metadata)
                
                # REQ-07: Handle tables semantically
                is_table = self._is_table_chunk(chunk)
                
                # REQ-NLP-01: Detect key claim sections
                key_claim_section = self._detect_key_claim_section(metadata)
                
                # Determine Hierarchy from Headings
                headings = metadata.get("headings", [])
                parent_chunk_id = None
                chunk_level = len(headings)
                
                # Create/Update Parent Sections
                if headings:
                    # The immediate parent is the section defined by the full headings path
                    parent_path = tuple(headings)
                    
                    # Ensure all ancestor sections exist
                    current_path = []
                    last_parent_id = None
                    
                    for h in headings:
                        current_path.append(h)
                        path_tuple = tuple(current_path)
                        
                        if path_tuple not in section_map:
                            section_map[path_tuple] = {
                                "id": str(uuid.uuid4()),
                                "content_parts": [],
                                "chunk_level": len(current_path) - 1, # 0-indexed levels for sections
                                "parent_chunk_id": last_parent_id,
                                "headings": list(path_tuple),
                                "is_reference": False # Default, updated if children are refs
                            }
                        
                        # Add this chunk's content to the section (aggregation)
                        section_map[path_tuple]["content_parts"].append(chunk.text)
                        
                        # Set reference flag if children are references
                        if is_reference:
                            section_map[path_tuple]["is_reference"] = True
                            
                        last_parent_id = section_map[path_tuple]["id"]
                    
                    # The leaf's parent is the deepest section
                    parent_chunk_id = section_map[parent_path]["id"]

                chunk_data = {
                    "id": chunk_uuid,
                    "parent_chunk_id": parent_chunk_id,
                    "chunk_level": chunk_level,
                    "content": chunk.text,
                    "chunk_index": i,
                    "metadata": metadata,
                    "is_reference": is_reference,
                    "is_table": is_table,
                }
                
                if key_claim_section:
                    chunk_data["metadata"]["key_claim_section"] = key_claim_section
                
                if is_table:
                    chunk_data["metadata"]["is_table"] = True
                    chunk_data["metadata"]["table_raw_content"] = chunk.text
                    chunk_data["content"] = chunk.text
                    chunk_data["embedding_content"] = f"[TABLE] {chunk.text[:500]}"
                    chunk_data["embedding_text"] = f"[TABLE] {chunk.text[:500]}"
                
                leaf_chunks.append(chunk_data)
            
            # Finalize Output List: Sections + Leaves
            final_chunks = []
            
            # Add Sections (Parents)
            # We sort by level so parents usually come before children, though not strictly required
            sorted_sections = sorted(section_map.items(), key=lambda item: len(item[0]))
            
            for path, sec in sorted_sections:
                # Construct combined content
                # We limit the size of parent chunks to avoid context overflow? 
                # For now, store full text. Retrieval matching will hit leaves, then fetch this parent.
                combined_content = "\n\n".join(sec["content_parts"])
                
                final_chunks.append({
                    "id": sec["id"],
                    "parent_chunk_id": sec["parent_chunk_id"],
                    "chunk_level": sec["chunk_level"],
                    "content": combined_content,
                    "chunk_index": -1, # Indicating it's a synthetic chunk
                    "metadata": {
                        "headings": sec["headings"],
                        "doc_item_type": "section",
                        "is_synthetic": True
                    },
                    "is_reference": sec["is_reference"],
                    "is_table": False
                })
                
            # Add Leaves
            final_chunks.extend(leaf_chunks)
                
            return final_chunks
        except Exception as e:
            print(f"Error parsing document {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return []

class ASTParser:
    """
    REQ-INGEST-01: Code-Aware Graphing
    
    Parses Python code using AST to extract:
    - Chunks: classes, methods, and functions as text segments
    - Graph data: nodes (Class, Method, Function) and edges (contains, inherits, calls)
    """
    
    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        Parses Python code using AST to split by classes and methods granularly.
        
        Returns:
            Dict with 'chunks' list and 'graph_data' dict containing nodes/edges.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
            
            tree = ast.parse(source)
            chunks = []
            chunk_index = 0
            
            # REQ-INGEST-01: Graph data structures
            nodes = []  # {"name": ..., "type": "Class"|"Method"|"Function"}
            edges = []  # {"source": ..., "target": ..., "relation": ..., "source_type": ..., "target_type": ...}
            
            lines = source.splitlines(keepends=True)
            
            def get_segment(node):
                # lineno is 1-indexed, end_lineno is inclusive
                if not hasattr(node, 'lineno') or not hasattr(node, 'end_lineno'):
                    return ""
                return "".join(lines[node.lineno-1 : node.end_lineno])
            
            def extract_calls(node, caller_name: str, caller_type: str):
                """Extract function/method calls from a node."""
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        # Get the called function name
                        if isinstance(child.func, ast.Name):
                            callee = child.func.id
                            edges.append({
                                "source": caller_name,
                                "target": callee,
                                "relation": "calls",
                                "source_type": caller_type,
                                "target_type": "Function"  # Assume function, may be class
                            })
                        elif isinstance(child.func, ast.Attribute):
                            # Method call like self.foo() or obj.bar()
                            callee = child.func.attr
                            edges.append({
                                "source": caller_name,
                                "target": callee,
                                "relation": "calls",
                                "source_type": caller_type,
                                "target_type": "Method"
                            })

            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                    
                    # REQ-INGEST-01: Add Class node
                    nodes.append({"name": class_name, "type": "Class"})
                    
                    # REQ-INGEST-01: Extract inheritance edges
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            parent_class = base.id
                            # Ensure parent class node exists
                            if not any(n["name"] == parent_class and n["type"] == "Class" for n in nodes):
                                nodes.append({"name": parent_class, "type": "Class"})
                            edges.append({
                                "source": class_name,
                                "target": parent_class,
                                "relation": "inherits",
                                "source_type": "Class",
                                "target_type": "Class"
                            })
                    
                    # 1. Capture Class Header / Docstring Context
                    class_doc = ast.get_docstring(node)
                    class_context = f"Parent Class: {class_name}\n"
                    if class_doc:
                        class_context += f"Class Docstring: {class_doc}\n"
                    
                    # Iterate methods
                    methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                    
                    for method in methods:
                        method_name = method.name
                        method_content = get_segment(method)
                        # Prepend context
                        full_content = f"{class_context}\n{method_content}"
                        
                        # REQ-INGEST-01: Add Method node and contains edge
                        nodes.append({"name": method_name, "type": "Method"})
                        edges.append({
                            "source": class_name,
                            "target": method_name,
                            "relation": "contains",
                            "source_type": "Class",
                            "target_type": "Method"
                        })
                        
                        # Extract calls from this method
                        extract_calls(method, method_name, "Method")
                        
                        chunks.append({
                            "content": full_content,
                            "chunk_index": chunk_index,
                            "type": "method",
                            "name": method_name,
                            "parent_class": class_name,
                            "metadata": {
                                "parent_class": class_name, 
                                "context_added": True,
                                "start_line": method.lineno,
                                "end_line": method.end_lineno
                            }
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
                            "metadata": {
                                "is_class_def": True,
                                "start_line": node.lineno,
                                "end_line": node.end_lineno
                            }
                        })
                        chunk_index += 1
                        
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Standalone function
                    func_name = node.name
                    content = get_segment(node)
                    
                    # REQ-INGEST-01: Add Function node
                    nodes.append({"name": func_name, "type": "Function"})
                    
                    # Extract calls from this function
                    extract_calls(node, func_name, "Function")
                    
                    chunks.append({
                        "content": content,
                        "chunk_index": chunk_index,
                        "type": "function",
                        "name": func_name,
                        "metadata": {
                            "start_line": node.lineno,
                            "end_line": node.end_lineno
                        }
                    })
                    chunk_index += 1
                else:
                    # Imports, constants, globals
                    # Optionally group them?
                    pass
            
            # REQ-INGEST-01: Return both chunks and graph data
            return {
                "chunks": chunks,
                "graph_data": {
                    "nodes": nodes,
                    "edges": edges
                }
            }

        except Exception as e:
            print(f"Error parsing code {file_path}: {e}")
            return {
                "chunks": [],
                "graph_data": {"nodes": [], "edges": []}
            }

