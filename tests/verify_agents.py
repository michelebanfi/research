import sys
import os
import asyncio
import time
import nest_asyncio

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Apply nest_asyncio
nest_asyncio.apply()

from src.ai_engine import AIEngine
from src.database import DatabaseClient
from src.agent import ResearchAgent
from src.agent_graph import run_agent_graph
from dotenv import load_dotenv

load_dotenv()

async def compare_agents():
    print("--- Starting Agent Comparison ---")
    try:
        db = DatabaseClient()
        ai = AIEngine()
        
        # Get project
        projects = db.get_projects()
        if not projects:
            print("❌ No projects found.")
            return
        
        project_id = projects[0]['id']
        project_name = projects[0]['name']
        print(f"Project: {project_name}")
        
        query = "What is the main topic of this knowledge base?"
        print(f"\nQuery: {query}")
        
        # 1. Test Standard Agent
        print("\n[1] Testing Standard ResearchAgent...")
        start_time = time.time()
        agent = ResearchAgent(ai, db, project_id)
        response = await agent.run(query)
        duration = time.time() - start_time
        
        print(f"✅ Standard Agent finished in {duration:.2f}s")
        print(f"Answer: {response.answer[:150]}...")
        if response.tools_used:
            print(f"Tools used: {response.tools_used}")
            
        # 2. Test LangGraph Agent
        print("\n[2] Testing LangGraph Agent...")
        start_time = time.time()
        # Re-initialize to avoid shared state issues if any
        try:
            lg_response = await run_agent_graph(query, ai, db, project_id)
            duration = time.time() - start_time
            print(f"✅ LangGraph Agent finished in {duration:.2f}s")
            print(f"Answer: {lg_response[:150]}...")
        except Exception as e:
            print(f"❌ LangGraph Agent failed: {e}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"❌ Setup failed: {e}")

if __name__ == "__main__":
    asyncio.run(compare_agents())
