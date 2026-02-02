import os
import ollama
from typing import List, Any

class AIEngine:
    def __init__(self):
        self.model = os.environ.get("OLLAMA_MODEL", "qwen:2.5")
        self.embed_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    def generate_embedding(self, text: str) -> List[float]:
        """Generates embedding for a given text."""
        try:
            response = ollama.embeddings(model=self.embed_model, prompt=text)
            return response["embedding"]
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []

    def generate_summary(self, text: str) -> str:
        """Generates a technical summary of the text."""
        try:
            prompt = f"Summarize the following technical content concisely:\n\n{text[:4000]}" # Truncate to avoid context window issues
            response = ollama.generate(model=self.model, prompt=prompt)
            return response["response"]
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Summary generation failed."

    def extract_keywords(self, text: str) -> List[str]:
        """Extracts keywords from the text."""
        try:
            prompt = f"Extract 5-10 technical keywords from the following text. Return them as a comma-separated list, nothing else:\n\n{text[:4000]}"
            response = ollama.generate(model=self.model, prompt=prompt)
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
            ranked_results = ranker.rank(rerankrequest)
            
            # Map back scores to original results
            # FlashRank returns list of formatted dicts with 'id', 'text', 'score', 'meta'
            # We need to preserve the original metadata from 'results'
            
            # Create a map of id -> result
            # Note: We synthesized IDs above if missing. 
            # Ideally results from DB have 'id'.
            
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
