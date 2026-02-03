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

    def _clean_json_string(self, json_str: str) -> str:
        """Cleans JSON string from Markdown code blocks and trailing commas."""
        import re
        # Remove markdown code blocks
        json_str = re.sub(r'```json\s*', '', json_str)
        json_str = re.sub(r'```\s*', '', json_str)
        
        # Remove trailing commas in arrays and objects (simple regex approach)
        # This is a basic heuristic: , followed by newline/space and closing bracket/brace
        json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
        
        return json_str.strip()

    def _normalize_graph_entities(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalizes entity names to lowercase to prevent duplicates."""
        if not data:
            return {"nodes": [], "edges": []}
            
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        
        # Normalize nodes
        for node in nodes:
            if "name" in node:
                node["name"] = node["name"].strip().lower()
                
        # Normalize edges
        for edge in edges:
            if "source" in edge:
                edge["source"] = edge["source"].strip().lower()
            if "target" in edge:
                edge["target"] = edge["target"].strip().lower()
                
        return {"nodes": nodes, "edges": edges}

    async def extract_metadata_graph_async(self, text: str) -> Dict[str, Any]:
        """
        Extracts entities (nodes) and relationships (edges) from the text asynchronously.
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
            
            # Robust JSON parsing
            import json
            import re
            
            clean_content = self._clean_json_string(content)
            
            # Try to find JSON object bounds if there's still extra text
            start = clean_content.find('{')
            end = clean_content.rfind('}') + 1
            if start != -1 and end != -1:
                clean_content = clean_content[start:end]
            
            try:
                data = json.loads(clean_content)
            except json.JSONDecodeError:
                # Fallback: try json_repair if available, or just fail gracefully
                # Since we don't have json_repair installed by default, we rely on the regex cleaning above.
                print(f"JSON Decode Error. Raw content: {content[:100]}...")
                return {"nodes": [], "edges": []}

            # Entity Resolution
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
