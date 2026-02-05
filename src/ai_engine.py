import os
import ollama
import asyncio
from typing import List, Any
from src.config import Config
from typing import Dict

class AIEngine:
    def __init__(self):
        self.model = Config.OLLAMA_MODEL
        self.embed_model = Config.OLLAMA_EMBED_MODEL
        self.client = ollama.Client()
        self.async_client = ollama.AsyncClient()

    def generate_embedding(self, text: str) -> List[float]:
        """Generates embedding for a given text."""
        try:
            response = self.client.embeddings(model=self.embed_model, prompt=text)
            return response["embedding"]
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []

    async def generate_embedding_async(self, text: str) -> List[float]:
        """Generates embedding for a given text asynchronously."""
        try:
            response = await self.async_client.embeddings(model=self.embed_model, prompt=text)
            return response["embedding"]
        except Exception as e:
            print(f"Error generating embedding async: {e}")
            return []

    async def generate_summary_async(self, text: str) -> str:
        """Generates a technical summary of the text asynchronously. Uses map-reduce for long texts."""
        MAX_CHUNK_SIZE = 4000
        
        try:
            if len(text) <= MAX_CHUNK_SIZE:
                prompt = f"Summarize the following technical content concisely:\n\n{text}"
                response = await self.async_client.generate(model=self.model, prompt=prompt)
                return response["response"]
            
            # Map Step: Split and summarize chunks
            chunks = [text[i:i+MAX_CHUNK_SIZE] for i in range(0, len(text), MAX_CHUNK_SIZE)]
            
            # Process chunks in parallel using asyncio.gather
            tasks = []
            for chunk in chunks:
                prompt = f"Summarize the following technical content concisely:\n\n{chunk}"
                tasks.append(self.async_client.generate(model=self.model, prompt=prompt))
            
            responses = await asyncio.gather(*tasks)
            chunk_summaries = [r["response"] for r in responses]
            
            # Reduce Step: Summarize the combined summaries
            combined_summary = "\n\n".join(chunk_summaries)
            # Recursively call generate_summary_async using await
            return await self.generate_summary_async(combined_summary)

        except Exception as e:
            print(f"Error generating summary async: {e}")
            return "Summary generation failed."

    # Keep synchronous version for backward compatibility if needed, but we'll try to move to async
    def generate_summary(self, text: str) -> str:
        """Synchronous wrapper for generate_summary_async."""
        return asyncio.run(self.generate_summary_async(text))

    # REQ-06: Synonym dictionary for common entity variations
    ENTITY_SYNONYMS = {
        # Key = canonical form (lowercase), Value = list of synonyms
        "large language model": ["llm", "llms", "large language models", "language model"],
        "machine learning": ["ml", "machine-learning"],
        "artificial intelligence": ["ai", "a.i."],
        "natural language processing": ["nlp", "n.l.p."],
        "convolutional neural network": ["cnn", "cnns", "convnet"],
        "recurrent neural network": ["rnn", "rnns"],
        "transformer": ["transformers", "transformer model", "transformer architecture"],
        "retrieval augmented generation": ["rag"],
        "knowledge graph": ["kg", "knowledge graphs"],
        "graph neural network": ["gnn", "gnns"],
        "deep learning": ["dl", "deep-learning"],
    }
    
    def _clean_json_string(self, json_str: str) -> str:
        """
        REQ-05: Cleans JSON string from Markdown code blocks and common LLM artifacts.
        Still used as a pre-processing step before Pydantic validation.
        """
        import re
        # Remove markdown code blocks
        json_str = re.sub(r'```json\s*', '', json_str)
        json_str = re.sub(r'```\s*', '', json_str)
        
        # Remove trailing commas in arrays and objects
        json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
        
        # Remove common LLM artifacts
        json_str = re.sub(r'^[^{]*', '', json_str)  # Remove text before first {
        json_str = re.sub(r'[^}]*$', '', json_str)  # Remove text after last }
        
        return json_str.strip()
    
    def _parse_graph_data_with_pydantic(self, json_str: str) -> Dict[str, Any]:
        """
        REQ-05: Parse and validate graph data using Pydantic.
        Falls back gracefully if Pydantic isn't available or validation fails.
        """
        import json
        
        # First clean the string
        clean_str = self._clean_json_string(json_str)
        
        try:
            from pydantic import BaseModel, validator
            from typing import List, Optional
            
            class GraphNode(BaseModel):
                name: str
                type: str = "Concept"
                
                @validator('name', pre=True, always=True)
                def clean_name(cls, v):
                    return str(v).strip() if v else ""
            
            class GraphEdge(BaseModel):
                source: str
                target: str
                relation: str = "related_to"
                
                @validator('source', 'target', pre=True, always=True)
                def clean_entity(cls, v):
                    return str(v).strip() if v else ""
            
            class GraphData(BaseModel):
                nodes: List[GraphNode] = []
                edges: List[GraphEdge] = []
            
            # Parse JSON first
            raw_data = json.loads(clean_str)
            
            # Validate with Pydantic
            validated = GraphData(**raw_data)
            
            return {
                "nodes": [n.dict() for n in validated.nodes],
                "edges": [e.dict() for e in validated.edges]
            }
            
        except ImportError:
            # Pydantic not available, use basic JSON parsing
            print("Pydantic not available, using basic JSON parsing")
            try:
                return json.loads(clean_str)
            except json.JSONDecodeError as e:
                print(f"JSON parse error: {e}")
                return {"nodes": [], "edges": []}
                
        except Exception as e:
            print(f"Pydantic validation error: {e}")
            # Try basic JSON as fallback
            try:
                return json.loads(clean_str)
            except:
                return {"nodes": [], "edges": []}

    def _resolve_entity_synonyms(self, name: str) -> str:
        """
        REQ-06: Resolve common synonyms to canonical forms.
        E.g., 'LLM' -> 'large language model'
        """
        name_lower = name.strip().lower()
        
        # Check if name matches any synonym
        for canonical, synonyms in self.ENTITY_SYNONYMS.items():
            if name_lower == canonical or name_lower in synonyms:
                return canonical
        
        return name_lower  # Return lowercase if no synonym match
    
    async def _resolve_entity_with_llm(self, entities: List[str]) -> Dict[str, str]:
        """
        REQ-06: Use LLM to resolve ambiguous entities to canonical forms.
        Returns mapping of original -> canonical name.
        """
        if not entities:
            return {}
        
        try:
            # First apply synonym dictionary
            resolved = {}
            ambiguous = []
            
            for entity in entities:
                canonical = self._resolve_entity_synonyms(entity)
                if canonical != entity.strip().lower():
                    resolved[entity] = canonical
                else:
                    ambiguous.append(entity)
            
            # For remaining ambiguous entities, use LLM if there are duplicates-ish
            # Only call LLM if we have multiple similar-looking entities
            if len(ambiguous) > 5:
                # Group potentially similar entities and ask LLM
                prompt = f"""Given these entity names extracted from technical documents, identify any that refer to the same concept and should be merged.
Return a JSON object where keys are the original names and values are the canonical (preferred) name to use.
Only include entities that should be merged. If an entity is unique, don't include it.

Entities: {ambiguous[:20]}

Return ONLY valid JSON like: {{"original1": "canonical1", "original2": "canonical1"}}
If no merges needed, return: {{}}"""
                
                response = await self.async_client.generate(model=self.model, prompt=prompt)
                
                try:
                    import json
                    llm_mappings = json.loads(self._clean_json_string(response["response"]))
                    resolved.update(llm_mappings)
                except:
                    pass  # LLM response wasn't valid JSON, skip
            
            return resolved
        except Exception as e:
            print(f"Error in LLM entity resolution: {e}")
            return {}

    def _normalize_graph_entities(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        REQ-06: Normalizes entity names using synonym dictionary.
        LLM resolution is done separately during ingestion for async support.
        """
        if not data:
            return {"nodes": [], "edges": []}
            
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        
        # Build name mapping from synonym resolution
        name_mapping = {}
        
        # First pass: resolve all names using synonym dictionary
        for node in nodes:
            if "name" in node:
                original = node["name"]
                canonical = self._resolve_entity_synonyms(original)
                name_mapping[original.strip().lower()] = canonical
                node["name"] = canonical
        
        # Apply same mapping to edges
        for edge in edges:
            if "source" in edge:
                original = edge["source"].strip().lower()
                edge["source"] = name_mapping.get(original, self._resolve_entity_synonyms(edge["source"]))
            if "target" in edge:
                original = edge["target"].strip().lower()
                edge["target"] = name_mapping.get(original, self._resolve_entity_synonyms(edge["target"]))
        
        # Deduplicate nodes with same name
        seen_names = {}
        unique_nodes = []
        for node in nodes:
            if node["name"] not in seen_names:
                seen_names[node["name"]] = True
                unique_nodes.append(node)
                
        return {"nodes": unique_nodes, "edges": edges}

    async def extract_metadata_graph_async(self, text: str) -> Dict[str, Any]:
        """
        REQ-05: Extracts entities (nodes) and relationships (edges) from the text asynchronously.
        Uses Pydantic validation for robust JSON parsing.
        Returns a dict with 'nodes' and 'edges'.
        """
        try:
            prompt = (
                "Analyze the following technical content and extract a Knowledge Graph.\n"
                "Focus on TECHNICAL CONCEPTS, TOOLS, SYSTEMS, and ALGORITHMS.\n"
                "IGNORE Authors, Dates, Institutions, and generic terms.\n"
                "Identify key entities (Nodes) and their relationships (Edges).\n"
                "Return ONLY a JSON object with this structure:\n"
                "{\n"
                '  "nodes": [{"name": "Entity Name", "type": "Concept|Tool|Metric|System|Person"}],\n'
                '  "edges": [{"source": "Entity Name", "target": "Entity Name", "relation": "relationship_type"}]\n'
                "}\n"
                "Ensure the JSON is valid and contains no other text.\n\n"
                f"{text[:4000]}"
            )
            response = await self.async_client.generate(model=self.model, prompt=prompt)
            content = response["response"]
            
            # REQ-05: Use Pydantic-based parsing for robust validation
            data = self._parse_graph_data_with_pydantic(content)
            
            # REQ-06: Apply entity normalization with synonym resolution
            data = self._normalize_graph_entities(data)
            return data
            
        except Exception as e:
            print(f"Error extracting graph metadata async: {e}")
            return {"nodes": [], "edges": []}

    def extract_metadata_graph(self, text: str) -> Dict[str, Any]:
        """Synchronous wrapper for extract_metadata_graph_async."""
        return asyncio.run(self.extract_metadata_graph_async(text))

    def rerank_results(self, query: str, results: List[Any], top_k: int = 5) -> List[Any]:
        """
        Re-ranks vector search results using FlashRank.
        results: List of dicts/objects containing 'content' and 'similarity'.
        Returns top_k re-ranked results.
        """
        try:
            from flashrank import Ranker, RerankRequest
            
            if not results:
                return []
            
            # Using a small, fast model. It will download on first run.
            ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="./.cache")
            
            # Prepare passages
            passages = [
                {"id": res.get("id", str(i)), "text": res.get("content")} 
                for i, res in enumerate(results)
            ]
            
            rerankrequest = RerankRequest(query=query, passages=passages)
            ranked_results = ranker.rerank(rerankrequest)
            
            id_to_result = {}
            for i, res in enumerate(results):
                rid = res.get("id", str(i))
                id_to_result[rid] = res
            
            final_results = []
            for item in ranked_results:
                rid = item['id']
                if rid in id_to_result:
                    original_res = id_to_result[rid]
                    original_res['rerank_score'] = item['score']
                    final_results.append(original_res)
            
            return final_results[:top_k]

        except Exception as e:
            print(f"Error re-ranking with FlashRank: {e}")
            return results[:top_k] # Fallback

    async def chat_with_context_async(self, query: str, context_chunks: List[Dict[str, Any]], chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Generates a response using RAG (Retrieval Augmented Generation).
        
        Args:
            query: The user's question
            context_chunks: List of dicts with 'content', 'similarity', optionally 'rerank_score'
            chat_history: Optional list of {'role': 'user'|'assistant', 'content': '...'}
            
        Returns:
            Dict with 'response' and optionally 'thinking'
        """
        try:
            # Build context from chunks with source attribution
            context_parts = []
            for i, chunk in enumerate(context_chunks):
                score = chunk.get('rerank_score', chunk.get('similarity', 0))
                context_parts.append(f"[Source {i+1}] (relevance: {score:.2f}):\n{chunk['content']}")
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Build conversation history if provided
            history_text = ""
            if chat_history:
                history_parts = []
                for msg in chat_history[-5:]:  # Last 5 messages for context
                    role = "User" if msg['role'] == 'user' else "Assistant"
                    history_parts.append(f"{role}: {msg['content']}")
                history_text = "\n".join(history_parts)
                history_text = f"\n\nCONVERSATION HISTORY:\n{history_text}\n"
            
            prompt = f"""You are a helpful research assistant with access to a knowledge base. Answer the user's question based on the provided context.

CONTEXT FROM KNOWLEDGE BASE:
{context}
{history_text}
USER QUESTION: {query}

Instructions:
- Answer based primarily on the provided context
- If the context doesn't contain enough information, clearly state that
- Cite sources when relevant using [Source N] notation
- Be concise but comprehensive
- If you need to reason through the answer, do so step by step"""

            response = await self.async_client.generate(model=self.model, prompt=prompt)
            
            return {
                "response": response["response"],
                "model": self.model
            }
            
        except Exception as e:
            print(f"Error in chat_with_context_async: {e}")
            return {
                "response": f"I encountered an error while processing your question: {str(e)}",
                "error": True
            }

    def chat_with_context(self, query: str, context_chunks: List[Dict[str, Any]], chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Synchronous wrapper for chat_with_context_async."""
        return asyncio.run(self.chat_with_context_async(query, context_chunks, chat_history))
