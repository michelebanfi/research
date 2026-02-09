import os
import json
import ollama
import asyncio
import re
from typing import List, Any, Dict
from src.config import Config


class AIEngine:
    def __init__(self):
        self.model = Config.OLLAMA_MODEL
        self.embed_model = Config.OLLAMA_EMBED_MODEL
        self.client = ollama.Client()
        self.async_client = ollama.AsyncClient()
        # REQ-IMP-03: Load synonyms dynamically from config file
        self._load_synonyms()
    
    def _load_synonyms(self):
        """
        REQ-IMP-03: Load entity synonyms from external JSON config.
        Allows runtime updates without code changes.
        """
        synonyms_path = os.path.join(os.path.dirname(__file__), '..', 'synonyms.json')
        if os.path.exists(synonyms_path):
            try:
                with open(synonyms_path, 'r') as f:
                    self.entity_synonyms = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load synonyms.json: {e}")
                self.entity_synonyms = self._get_default_synonyms()
        else:
            self.entity_synonyms = self._get_default_synonyms()
    
    def _get_default_synonyms(self) -> Dict[str, List[str]]:
        """Fallback synonyms if config file is missing."""
        return {
            "large language model": ["llm", "llms"],
            "machine learning": ["ml"],
            "artificial intelligence": ["ai"],
        }

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
    
    def _split_text_semantically(self, text: str, max_size: int) -> List[str]:
        """
        REQ-IMP-01: Split text on sentence boundaries, preserving complete sentences.
        Prevents cutting words/sentences in half during summarization.
        """
        # Split on sentence-ending punctuation followed by whitespace
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_size:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # Handle case where single sentence exceeds max_size
                if len(sentence) > max_size:
                    # Fall back to word-level splitting for very long sentences
                    words = sentence.split()
                    current_chunk = ""
                    for word in words:
                        if len(current_chunk) + len(word) + 1 <= max_size:
                            current_chunk += (" " if current_chunk else "") + word
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = word
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]

    async def generate_summary_async(self, text: str) -> str:
        """Generates a technical summary of the text asynchronously. Uses map-reduce for long texts."""
        MAX_CHUNK_SIZE = 4000
        
        try:
            if len(text) <= MAX_CHUNK_SIZE:
                prompt = f"Summarize the following technical content concisely:\n\n{text}"
                response = await self.async_client.generate(model=self.model, prompt=prompt)
                return response["response"]
            
            # REQ-IMP-01: Use semantic splitting instead of rigid character slicing
            chunks = self._split_text_semantically(text, MAX_CHUNK_SIZE)
            
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

    # Synchronous wrapper for generate_summary_async
    def generate_summary(self, text: str) -> str:
        """Synchronous wrapper for generate_summary_async."""
        return asyncio.run(self.generate_summary_async(text))
    
    async def extract_key_claims_async(self, text: str, section_type: str = "abstract") -> List[str]:
        """
        REQ-NLP-01: Extract key claims from abstract/conclusion sections.
        
        Uses LLM to identify the main claims, findings, or contributions.
        Returns list of bullet-point claims for metadata storage.
        
        Args:
            text: The section text to analyze
            section_type: Type of section ("abstract", "conclusion", "results")
            
        Returns:
            List of extracted claim strings
        """
        if not text or len(text.strip()) < 50:
            return []
        
        try:
            prompt = f"""Extract 3-5 key claims or findings from this {section_type}:

{text[:2000]}

Return as a JSON array of concise claim strings. Each claim should be a single sentence.
Example: ["This study proposes X", "Results show Y improves by Z%", "The approach outperforms baseline"]

Return ONLY the JSON array, no other text."""

            response = await self.async_client.generate(model=self.model, prompt=prompt)
            content = response["response"]
            
            # Parse JSON array
            clean_content = self._clean_json_string(content)
            # Handle case where response is wrapped differently
            if not clean_content.startswith("["):
                import re
                array_match = re.search(r'\[.*\]', clean_content, re.DOTALL)
                if array_match:
                    clean_content = array_match.group(0)
                else:
                    return []
            
            import json
            claims = json.loads(clean_content)
            
            # Validate it's a list of strings
            if isinstance(claims, list):
                return [str(c).strip() for c in claims if c and str(c).strip()][:5]
            return []
            
        except Exception as e:
            print(f"Error extracting key claims: {e}")
            return []
    
    # REQ-IMP-03: ENTITY_SYNONYMS now loaded dynamically from synonyms.json
    # See __init__ and _load_synonyms()
    
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
                source_type: str = "Concept"  # REQ-DATA-01: type for source node
                target_type: str = "Concept"  # REQ-DATA-01: type for target node
                
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
        REQ-IMP-03: Now uses dynamically loaded synonyms.
        E.g., 'LLM' -> 'large language model'
        """
        name_lower = name.strip().lower()
        
        # Check if name matches any synonym
        for canonical, synonyms in self.entity_synonyms.items():
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
            # REQ-DATA-01: Updated prompt to request source_type and target_type for edges
            prompt = (
                "Analyze the following technical content and extract a Knowledge Graph.\n"
                "Focus on TECHNICAL CONCEPTS, TOOLS, SYSTEMS, and ALGORITHMS.\n"
                "IGNORE Authors, Dates, Institutions, and generic terms.\n"
                "Identify key entities (Nodes) and their relationships (Edges).\n"
                "Return ONLY a JSON object with this structure:\n"
                "{\n"
                '  "nodes": [{"name": "Entity Name", "type": "Concept|Tool|Metric|System|Person"}],\n'
                '  "edges": [{"source": "Entity Name", "target": "Entity Name", "relation": "relationship_type", "source_type": "type of source", "target_type": "type of target"}]\n'
                "}\n"
                "IMPORTANT: Include source_type and target_type in each edge to disambiguate entities with the same name.\n"
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

    def _estimate_tokens(self, text: str) -> int:
        """
        REQ-STABILITY-01: Estimate token count for a text string.
        
        Uses a simple word-based estimation: ~1.3 tokens per word on average.
        This is a conservative estimate that works for most English text.
        """
        if not text:
            return 0
        # Rough estimation: split by whitespace, multiply by factor
        word_count = len(text.split())
        return int(word_count * 1.3)

    async def chat_with_context_async(self, query: str, context_chunks: List[Dict[str, Any]], chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Generates a response using RAG (Retrieval Augmented Generation).
        
        REQ-STABILITY-01: Automatically truncates context and history if exceeding token limit.
        
        Args:
            query: The user's question
            context_chunks: List of dicts with 'content', 'similarity', optionally 'rerank_score'
            chat_history: Optional list of {'role': 'user'|'assistant', 'content': '...'}
            
        Returns:
            Dict with 'response' and optionally 'thinking'
        """
        try:
            # REQ-STABILITY-01: Get token limit from config
            max_tokens = Config.MODEL_CONTEXT_LIMIT
            
            # Reserve tokens for system prompt, query, and response
            system_prompt_tokens = 150  # Approximate tokens for instructions
            query_tokens = self._estimate_tokens(query)
            response_reserve = 500  # Save space for model response
            available_tokens = max_tokens - system_prompt_tokens - query_tokens - response_reserve
            
            # Build context from chunks with source attribution, respecting token limit
            context_parts = []
            context_tokens = 0
            truncated_count = 0
            
            for i, chunk in enumerate(context_chunks):
                score = chunk.get('rerank_score', chunk.get('similarity', 0))
                chunk_text = f"[Source {i+1}] (relevance: {score:.2f}):\n{chunk['content']}"
                chunk_tokens = self._estimate_tokens(chunk_text)
                
                # Check if adding this chunk would exceed limit
                if context_tokens + chunk_tokens > available_tokens * 0.7:  # Use 70% for context
                    truncated_count += 1
                    continue
                
                context_parts.append(chunk_text)
                context_tokens += chunk_tokens
            
            if truncated_count > 0:
                print(f"REQ-STABILITY-01: Truncated {truncated_count} context chunks to fit token limit")
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Build conversation history if provided, respecting remaining token budget
            history_text = ""
            remaining_tokens = available_tokens - context_tokens
            
            if chat_history and remaining_tokens > 100:
                history_parts = []
                history_tokens = 0
                
                # Process from most recent, keeping as many as fit
                for msg in reversed(chat_history[-10:]):  # Consider last 10 messages
                    role = "User" if msg['role'] == 'user' else "Assistant"
                    msg_text = f"{role}: {msg['content']}"
                    msg_tokens = self._estimate_tokens(msg_text)
                    
                    if history_tokens + msg_tokens > remaining_tokens * 0.8:  # Use 80% of remaining
                        break
                    
                    history_parts.insert(0, msg_text)  # Insert at beginning to maintain order
                    history_tokens += msg_tokens
                
                if history_parts:
                    history_text = "\n".join(history_parts)
                    history_text = f"\n\nCONVERSATION HISTORY:\n{history_text}\n"
                
                if len(history_parts) < len(chat_history):
                    print(f"REQ-STABILITY-01: Truncated chat history from {len(chat_history)} to {len(history_parts)} messages")
            
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

    async def determine_intent(self, query: str) -> str:
        """
        Decides if the query requires a 'SPECIFIC' search or a 'BROAD' overview.
        """
        prompt = f"""
        Analyze the following user query: "{query}"
        
        Classify it into one of two categories:
        1. SPECIFIC: The user is asking for specific details, code definitions, specific facts, or concepts.
        2. BROAD: The user is asking for a high-level summary, an overview of what files exist, or what the project does.
        
        Return ONLY the word 'SPECIFIC' or 'BROAD'.
        """
        try:
            # Using a cheaper/faster model helps here if available, but your main model is fine
            response = await self.async_client.generate(model=self.model, prompt=prompt)
            intent = response["response"].strip().upper()
            # Fallback to SPECIFIC if response is messy
            return "BROAD" if "BROAD" in intent else "SPECIFIC"
        except Exception:
            return "SPECIFIC"
