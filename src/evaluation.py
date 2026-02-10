import asyncio
import time
import argparse
import sys
import json
from src.ai_engine import AIEngine
from src.database import DatabaseClient

async def evaluate_query(query: str, project_id: str, ai_engine: AIEngine, db: DatabaseClient):
    print(f"\nEvaluating query: '{query}'")
    results = {}
    
    # 1. Measure Embedding Latency
    start_time = time.time()
    query_embedding = await ai_engine.generate_embedding_async(query)
    embedding_time = time.time() - start_time
    results['embedding_time_ms'] = embedding_time * 1000
    print(f"  Embedding Latency: {embedding_time*1000:.2f} ms")
    
    # 2. Measure Retrieval Latency (Vector Search)
    start_time = time.time()
    # Note: Using match_count=20 to get candidates for re-ranking
    vector_results = db.search_vectors(query_embedding, match_threshold=0.3, project_id=project_id, match_count=20)
    retrieval_time = time.time() - start_time
    results['retrieval_time_ms'] = retrieval_time * 1000
    results['retrieved_count'] = len(vector_results)
    print(f"  Retrieval Latency: {retrieval_time*1000:.2f} ms (Found {len(vector_results)} chunks)")
    
    if not vector_results:
        print("  No results found. Skipping ranking/generation.")
        return results

    # 3. Measure Re-ranking Latency
    start_time = time.time()
    # Re-rank top 20, return top 5
    reranked_results = ai_engine.rerank_results(query, vector_results, top_k=5)
    rerank_time = time.time() - start_time
    results['rerank_time_ms'] = rerank_time * 1000
    print(f"  Re-ranking Latency: {rerank_time*1000:.2f} ms")
    
    # 4. Measure Generation Latency
    start_time = time.time()
    # Use chat_with_context_async
    response_data = await ai_engine.chat_with_context_async(query, reranked_results)
    generation_time = time.time() - start_time
    results['generation_time_ms'] = generation_time * 1000
    print(f"  Generation Latency: {generation_time*1000:.2f} ms")
    
    response_text = response_data.get('response', '')
    
    # 5. LLM-as-a-judge Evaluation
    # Context string for judge
    context_str = ""
    for i, res in enumerate(reranked_results[:2]):
        context_str += f"{i+1}. {res['content'][:200]}...\n"
        
    judge_prompt = f"""You are an expert evaluator. Rate the quality of the AI response to the user query on a scale of 1 to 5.

User Query: "{query}"

Retrieved Context (Top 2): 
{context_str}

AI Response: "{response_text}"

Criteria:
1 - Irrelevant or Hallucinated
3 - Partially Correct
5 - Excellent, Accurate, and Comprehensive

Return ONLY a JSON object: {{"score": <int>, "reason": "<string>"}}"""

    try:
        judge_response = await ai_engine._openrouter_generate(judge_prompt)
        clean_json = ai_engine._clean_json_string(judge_response)
        eval_metrics = json.loads(clean_json)
        results['relevance_score'] = eval_metrics.get('score')
        results['relevance_reason'] = eval_metrics.get('reason')
        print(f"  Relevance Score: {results['relevance_score']}/5")
        print(f"  Reason: {results['relevance_reason']}")
    except Exception as e:
        print(f"  Evaluation Failed: {e}")
        results['relevance_score'] = None
        results['relevance_reason'] = str(e)

    return results

async def main():
    parser = argparse.ArgumentParser(description="Evaluate Research Assistant Performance")
    parser.add_argument("--query", type=str, help="Single query to evaluate")
    parser.add_argument("--project_id", type=str, required=True, help="Project ID to search in")
    args = parser.parse_args()
    
    try:
        db = DatabaseClient()
        ai_engine = AIEngine()
        
        queries = []
        if args.query:
            queries = [args.query]
        else:
            # Default benchmark queries
            queries = [
                "Summarize the main findings",
                "What methodologies were used?",
                "List the key limitations mentioned"
            ]
            print("No --query provided, running default benchmark set.")
        
        print(f"Starting Evaluation on Project: {args.project_id}")
            
        metrics = []
        for q in queries:
            m = await evaluate_query(q, args.project_id, ai_engine, db)
            metrics.append(m)
            
        # calculate averages
        if metrics:
            generation_times = [m.get('generation_time_ms', 0) for m in metrics if m.get('generation_time_ms')]
            avg_gen_time = sum(generation_times) / len(generation_times) if generation_times else 0
            
            scores = [m.get('relevance_score', 0) for m in metrics if m.get('relevance_score')]
            avg_score = sum(scores) / len(scores) if scores else 0
            
            print("\n=== Summary ===")
            print(f"Average Generation Time: {avg_gen_time:.2f} ms")
            print(f"Average Relevance Score: {avg_score:.1f}/5")
        else:
            print("\nNo metrics collected.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    current_loop = asyncio.get_event_loop()
    current_loop.run_until_complete(main())
