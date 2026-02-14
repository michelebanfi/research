import os
import json
import ollama
import asyncio
import re
import hashlib
from functools import lru_cache
from typing import List, Any, Dict, Tuple, Optional, Callable
from openai import OpenAI, AsyncOpenAI
from src.config import Config
from src.cache import Cache

# Try to import tenacity for retry logic
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False
    print("Warning: tenacity not installed. Retry logic disabled. Install with: pip install tenacity")


class AIEngine:
    
    def __init__(self):
        # REQ-PERF-01: Initialize persistent cache
        self.cache = Cache()
        
        # LLM via OpenRouter
        self.model = Config.OPENROUTER_MODEL
        self.openai_client = OpenAI(
            base_url=Config.OPENROUTER_BASE_URL,
            api_key=Config.OPENROUTER_KEY,
        )
        self.async_openai_client = AsyncOpenAI(
            base_url=Config.OPENROUTER_BASE_URL,
            api_key=Config.OPENROUTER_KEY,
        )
        # Embeddings still via Ollama
        self.embed_model = Config.OLLAMA_EMBED_MODEL
        self.ollama_client = ollama.Client()
        self.ollama_async_client = ollama.AsyncClient()
        # REQ-IMP-03: Load synonyms dynamically from config file
        self._load_synonyms()
        
        # REQ-IMP-06: Rate limiting semaphore
        self._rate_limiter = asyncio.Semaphore(Config.API_MAX_CONCURRENT_CALLS)
    
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
        """Generates embedding for a given text via Ollama. Uses cache if available."""
        # REQ-IMP-08: Check cache first
        cached = self.cache.get("embedding", text)
        if cached:
            return cached
        
        try:
            response = self.ollama_client.embeddings(model=self.embed_model, prompt=text)
            embedding = response["embedding"]
            self.cache.set("embedding", text, embedding)
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []

    async def generate_embedding_async(self, text: str) -> List[float]:
        """Generates embedding for a given text asynchronously via Ollama. Uses cache if available."""
        # REQ-IMP-08: Check cache first
        cached = self.cache.get("embedding", text)
        if cached:
            return cached
        
        try:
            response = await self.ollama_async_client.embeddings(model=self.embed_model, prompt=text)
            embedding = response["embedding"]
            self.cache.set("embedding", text, embedding)
            return embedding
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

    async def _openrouter_generate(self, prompt: str, return_model_name: bool = False, response_format: dict = None) -> Any:
        """
        Helper to call OpenRouter via OpenAI SDK.
        REQ-IMP-06: Uses rate limiting semaphore.
        REQ-IMP-07: Uses retry logic with exponential backoff.
        
        Args:
            prompt: The user prompt.
            return_model_name: If True, returns (content, model_name). Else returns content.
            response_format: Optional dict for JSON mode (e.g. {"type": "json_object"}).
        """
        async with self._rate_limiter:
            if TENACITY_AVAILABLE:
                return await self._openrouter_generate_with_retry(prompt, return_model_name, response_format)
            else:
                return await self._openrouter_generate_simple(prompt, return_model_name, response_format)
    
    async def _openrouter_generate_simple(self, prompt: str, return_model_name: bool = False, response_format: dict = None) -> Any:
        """Simple implementation without tenacity retry."""
        last_error = None
        for attempt in range(Config.API_RETRY_ATTEMPTS):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}]
                }
                if response_format:
                    kwargs["response_format"] = response_format
                
                response = await self.async_openai_client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""
                if return_model_name:
                    return content, response.model
                return content
            except Exception as e:
                last_error = e
                if attempt < Config.API_RETRY_ATTEMPTS - 1:
                    wait_time = min(Config.API_RETRY_MIN_WAIT_S * (2 ** attempt), Config.API_RETRY_MAX_WAIT_S)
                    await asyncio.sleep(wait_time)
        raise last_error or Exception("Max retries exceeded")
    
    async def _openrouter_generate_with_retry(self, prompt: str, return_model_name: bool = False, response_format: dict = None) -> Any:
        """Implementation with tenacity retry decorator."""
        @retry(
            stop=stop_after_attempt(Config.API_RETRY_ATTEMPTS),
            wait=wait_exponential(min=Config.API_RETRY_MIN_WAIT_S, max=Config.API_RETRY_MAX_WAIT_S)
        )
        async def _call():
            kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}]
            }
            if response_format:
                kwargs["response_format"] = response_format

            response = await self.async_openai_client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            if return_model_name:
                return content, response.model
            return content
        return await _call()

    def _openrouter_generate_sync(self, prompt: str) -> str:
        """
        Synchronous helper to call OpenRouter via OpenAI SDK.
        REQ-IMP-07: Uses simple retry logic.
        """
        last_error = None
        for attempt in range(Config.API_RETRY_ATTEMPTS):
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                last_error = e
                if attempt < Config.API_RETRY_ATTEMPTS - 1:
                    import time
                    wait_time = min(Config.API_RETRY_MIN_WAIT_S * (2 ** attempt), Config.API_RETRY_MAX_WAIT_S)
                    time.sleep(wait_time)
        raise last_error or Exception("Max retries exceeded")

    async def generate_summary_async(self, text: str, _depth: int = 0) -> str:
        """Generates a technical summary of the text asynchronously. Uses map-reduce for long texts."""
        MAX_CHUNK_SIZE = 4000
        MAX_RECURSION_DEPTH = 3
        
        try:
            if _depth >= MAX_RECURSION_DEPTH:
                # Force single-pass summary on truncated text to prevent infinite recursion
                prompt = f"Summarize the following technical content concisely:\n\n{text[:MAX_CHUNK_SIZE]}"
                return await self._openrouter_generate(prompt)
            
            if len(text) <= MAX_CHUNK_SIZE:
                prompt = f"Summarize the following technical content concisely:\n\n{text}"
                return await self._openrouter_generate(prompt)
            
            # REQ-IMP-01: Use semantic splitting instead of rigid character slicing
            chunks = self._split_text_semantically(text, MAX_CHUNK_SIZE)
            
            # Process chunks in parallel using asyncio.gather
            tasks = []
            for chunk in chunks:
                prompt = f"Summarize the following technical content concisely:\n\n{chunk}"
                tasks.append(self._openrouter_generate(prompt))
            
            responses = await asyncio.gather(*tasks)
            chunk_summaries = responses  # Already strings from _openrouter_generate
            
            # Reduce Step: Summarize the combined summaries
            combined_summary = "\n\n".join(chunk_summaries)
            # Recursively call generate_summary_async with incremented depth
            return await self.generate_summary_async(combined_summary, _depth + 1)

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

            content = await self._openrouter_generate(prompt)
            
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
        
        # Remove common LLM artifacts — handle both JSON objects {...} and arrays [...]
        json_str = json_str.strip()
        first_brace = json_str.find('{')
        first_bracket = json_str.find('[')
        
        if first_brace == -1 and first_bracket == -1:
            return json_str  # No JSON structure found, return as-is
        elif first_brace == -1:
            start, end_char = first_bracket, ']'
        elif first_bracket == -1:
            start, end_char = first_brace, '}'
        else:
            start = min(first_brace, first_bracket)
            end_char = '}' if start == first_brace else ']'
        
        end = json_str.rfind(end_char)
        if end > start:
            json_str = json_str[start:end + 1]
        
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
            from pydantic import BaseModel, field_validator
            from typing import List, Optional
            
            class GraphNode(BaseModel):
                name: str
                type: str = "Concept"
                
                @field_validator('name', mode='before')
                @classmethod
                def clean_name(cls, v):
                    return str(v).strip() if v else ""
            
            class GraphEdge(BaseModel):
                source: str
                target: str
                relation: str = "related_to"
                source_type: str = "Concept"  # REQ-DATA-01: type for source node
                target_type: str = "Concept"  # REQ-DATA-01: type for target node
                
                @field_validator('source', 'target', mode='before')
                @classmethod
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
                "nodes": [n.model_dump() for n in validated.nodes],
                "edges": [e.model_dump() for e in validated.edges]
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
        Force lowercase to prevent duplicates.
        """
        name_lower = name.strip().lower()
        
        # Check if name matches any synonym
        for canonical, synonyms in self.entity_synonyms.items():
            # Ensure canonical is also compared/returned in lowercase
            canonical_lower = canonical.lower()
            if name_lower == canonical_lower or name_lower in [s.lower() for s in synonyms]:
                return canonical_lower
        
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
                
                content = await self._openrouter_generate(prompt)
                
                try:
                    import json
                    llm_mappings = json.loads(self._clean_json_string(content))
                    resolved.update(llm_mappings)
                except:
                    pass  # LLM response wasn't valid JSON, skip
            
            return resolved
        except Exception as e:
            print(f"Error in LLM entity resolution: {e}")
            return {}

    def _normalize_graph_entities(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        REQ-06: Normalizes entity names using synonym dictionary and case-insensitive matching.
        Improves upon simple lowercasing by preserving display case using frequency or canonical mapping.
        """
        if not data:
            return {"nodes": [], "edges": []}
            
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        
        # 1. First pass: Apply explicit synonyms from config
        # Name -> Canonical Name (all lower for keying)
        name_mapping = {} 
        
        for node in nodes:
            original_name = node["name"]
            # Check explicit synonyms first
            canonical_synonym = self._resolve_entity_synonyms(original_name)
            
            # If synonym found (and it's different), map it
            if canonical_synonym != original_name.lower():
                name_mapping[original_name.lower()] = canonical_synonym
            else:
                # Otherwise map to itself (lower) for now
                name_mapping[original_name.lower()] = original_name.lower()

        # 2. Second pass: Identify "Best Display Name" for each canonical lower key
        # e.g. "LLM", "llm", "Llm" -> all map to "llm" key, but we want "LLM" for display
        
        # key: lower_name, value: Counter(original_names)
        from collections import Counter
        display_candidates = {}
        
        for node in nodes:
            original = node["name"]
            lower_key = name_mapping.get(original.lower(), original.lower())
            
            if lower_key not in display_candidates:
                display_candidates[lower_key] = Counter()
            display_candidates[lower_key][original] += 1
            
        # Determine best display name for each key
        final_display_names = {}
        for key, counter in display_candidates.items():
            # Heuristic: Prefer all-caps if short (e.g. LLM), otherwise most frequent
            # If there's a synonym override in config (canonical), use that as display if it looks good
            
            # Check if key itself was a canonical from synonym dict that might not be in the counter
            # In _resolve_entity_synonyms we return lower(), so we might want to title case it if it's not an acronym
            best_name = counter.most_common(1)[0][0]
            
            # If the key is less than 4 chars and we have an uppercase version, prefer it
            if len(key) < 5:
                for cand in counter:
                    if cand.isupper():
                        best_name = cand
                        break
            
            final_display_names[key] = best_name

        # 3. Apply normalization
        unique_nodes = {}
        
        for node in nodes:
            original = node["name"]
            lower_key = name_mapping.get(original.lower(), original.lower())
            display_name = final_display_names.get(lower_key, original)
            
            # Create/Update node
            # We merge types if we see duplicates (maybe?) but for now just take the first or "Concept"
            if display_name not in unique_nodes:
                node["name"] = display_name
                unique_nodes[display_name] = node
            else:
                # Merge logic: if existing is "Concept" and new is specific, take specific
                existing = unique_nodes[display_name]
                if existing.get("type") == "Concept" and node.get("type") != "Concept":
                    existing["type"] = node.get("type")

        # Rewrite edges
        final_edges = []
        for edge in edges:
            s_original = edge.get("source", "")
            t_original = edge.get("target", "")
            
            s_key = name_mapping.get(s_original.lower(), s_original.lower())
            t_key = name_mapping.get(t_original.lower(), t_original.lower())
            
            s_final = final_display_names.get(s_key, s_original)
            t_final = final_display_names.get(t_key, t_original)
            
            if s_final and t_final and s_final != t_final:
                edge["source"] = s_final
                edge["target"] = t_final
                final_edges.append(edge)
                
        return {"nodes": list(unique_nodes.values()), "edges": final_edges}

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
            content = await self._openrouter_generate(prompt)
            
            # REQ-05: Use Pydantic-based parsing for robust validation
            data = self._parse_graph_data_with_pydantic(content)
            
            # REQ-06: Apply entity normalization with synonym resolution
            data = self._normalize_graph_entities(data)
            return data
            
        except Exception as e:
            print(f"Error extracting graph metadata async: {e}")
            return {"nodes": [], "edges": []}

    async def expand_query(self, query: str, num_variations: int = 3) -> List[str]:
        """
        REQ-IMP-09: Generates alternative search queries to improve retrieval recall.
        Uses LLM to find synonyms and related technical terms.
        """
        try:
            prompt = f"""Generate {num_variations} alternative search queries for the following user question to improve information retrieval from a technical knowledge base.
    Focus on synonyms, technical terms, and alternative phrasings.
    
    User Question: "{query}"
    
    Return ONLY a JSON array of strings. Example: ["query variation 1", "query variation 2", "third variation"]"""
            
            content = await self._openrouter_generate(prompt)
            clean_content = self._clean_json_string(content)
            
            import json
            variations = json.loads(clean_content)
            
            if isinstance(variations, list):
                # Return original + variations, unique
                combined = [query] + [str(v) for v in variations if v]
                return list(set(combined))
            return [query]
            
        except Exception as e:
            print(f"Error expanding query: {e}")
            return [query]

    def extract_metadata_graph(self, text: str) -> Dict[str, Any]:
        """Synchronous wrapper for extract_metadata_graph_async."""
        return asyncio.run(self.extract_metadata_graph_async(text))


    def _merge_search_results(self, all_results: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Merges results from multiple queries using a simplified Reciprocal Rank Fusion.
        """
        fused_scores = {}
        doc_map = {}
        
        # RRF constant
        k = 60
        
        for result_set in all_results:
            for rank, doc in enumerate(result_set):
                doc_id = doc['id']
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc
                
                # RRF score: 1 / (k + rank)
                score = 1.0 / (k + rank + 1)
                
                if doc_id in fused_scores:
                    fused_scores[doc_id] += score
                else:
                    fused_scores[doc_id] = score
        
        # Sort by fused score
        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        
        final_results = []
        for doc_id in sorted_ids:
            doc = doc_map[doc_id]
            doc['rrf_score'] = fused_scores[doc_id]
            final_results.append(doc)
            
        return final_results

    async def retrieve_advanced(self, query: str, db_client: Any, project_id: str, limit: int = 10, callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> List[Dict[str, Any]]:
        """
        Orchestrates advanced retrieval:
        1. Expand query
        2. Parallel search for all variations
        3. Merge results (RRF)
        4. Rerank
        """
        if callback:
             callback({"type": "search", "content": f"Exploring knowledge base for: '{query}'", "metadata": {"stage": "init"}})
        
        # 1. Expand Query
        queries = await self.expand_query(query)
        print(f"Expanded queries: {queries}")
        
        if callback and len(queries) > 1:
             callback({"type": "search", "content": f"Expanded terms: {', '.join(queries[1:])}", "metadata": {"stage": "expansion", "queries": queries}})
        
        # 2. Parallel Search
        search_tasks = []
        for q in queries:
            async def search_single(q_str):
                # Generate embedding
                embedding = await self.generate_embedding_async(q_str)
                # Search DB (using synchronous client in async wrapper if needed)
                # Note: db_client.search_vectors is sync. We should run it in executor or similar if blocking
                # For now, running sync in loop
                return db_client.search_vectors(embedding, match_threshold=0.3, project_id=project_id, match_count=limit*2)
            
            search_tasks.append(search_single(q))
            
        all_results = await asyncio.gather(*search_tasks)
        
        # 3. Merge Results
        merged_results = self._merge_search_results(all_results)
        
        if callback:
             callback({"type": "search", "content": f"Found {len(merged_results)} potential matches from multiple queries.", "metadata": {"stage": "merge", "count": len(merged_results)}})
        
        # 4. Rerank top candidates
        # Take top 3x limit for reranking
        candidates = merged_results[:limit*3]
        final_results = self.rerank_results(query, candidates, top_k=limit)
        
        if callback:
             callback({"type": "search", "content": f"Top {len(final_results)} most relevant documents selected.", "metadata": {"stage": "rerank", "top_k": len(final_results)}})
        
        return final_results
        
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
            
            # Use configured model
            model_name = Config.RERANK_MODEL_NAME
            # Note: FlashRank automatically uses MPS if available on Mac M1/M2/M3
            ranker = Ranker(model_name=model_name, cache_dir="./.cache")
            
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

            response_text, model_name = await self._openrouter_generate(prompt, return_model_name=True)
            
            return {
                "response": response_text,
                "model": model_name
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
            response_text = await self._openrouter_generate(prompt)
            intent = response_text.strip().upper()
            # Fallback to SPECIFIC if response is messy
            return "BROAD" if "BROAD" in intent else "SPECIFIC"
        except Exception:
            return "SPECIFIC"
