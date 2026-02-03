import os
import ollama
import asyncio
from typing import List, Any
from src.config import Config

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

    def generate_summary(self, text: str) -> str:
        """Generates a technical summary of the text. Uses map-reduce for long texts."""
        MAX_CHUNK_SIZE = 4000
        
        try:
            if len(text) <= MAX_CHUNK_SIZE:
                prompt = f"Summarize the following technical content concisely:\n\n{text}"
                response = self.client.generate(model=self.model, prompt=prompt)
                return response["response"]
            
            # Map Step: Split and summarize chunks
            chunks = [text[i:i+MAX_CHUNK_SIZE] for i in range(0, len(text), MAX_CHUNK_SIZE)]
            chunk_summaries = []
            for chunk in chunks:
                # Recurse or call generate for chunk
                # Ideally we want to process these in parallel if using async, but here we are in sync method
                # We can just call self.generate_summary recursively which handles the splitting if chunk is still too big (unlikely with this split)
                # or just call client.generate if we are sure it fits.
                # Let's call client.generate to be efficient for this level.
                prompt = f"Summarize the following technical content concisely:\n\n{chunk}"
                response = self.client.generate(model=self.model, prompt=prompt)
                chunk_summaries.append(response["response"])
            
            # Reduce Step: Summarize the combined summaries
            combined_summary = "\n\n".join(chunk_summaries)
            # Recursively call generate_summary in case the combined summary is still huge
            return self.generate_summary(combined_summary)

        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Summary generation failed."

    def extract_keywords(self, text: str) -> List[str]:
        """Extracts keywords from the text."""
        try:
            prompt = f"Extract 5-10 technical keywords from the following text. Return them as a comma-separated list, nothing else:\n\n{text[:4000]}"
            response = self.client.generate(model=self.model, prompt=prompt)
            keywords_str = response["response"]
            keywords = [k.strip() for k in keywords_str.split(",")]
            return keywords
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            return []

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
