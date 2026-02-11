import sys
import os
import asyncio
import nest_asyncio

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Apply nest_asyncio
nest_asyncio.apply()

from src.ai_engine import AIEngine
from src.database import DatabaseClient
from src.agent_graph import run_agent_graph
from dotenv import load_dotenv

load_dotenv()

async def test_langgraph():
    print("Initializing components...")
    try:
        db = DatabaseClient()
        ai = AIEngine()
        
        # Use a dummy project ID or fetch one
        projects = db.get_projects()
        if not projects:
            print("❌ No projects found to test.")
            return
        
        project_id = projects[0]['id']
        print(f"Using project: {projects[0]['name']} ({project_id})")
        
        query = "What is this knowledge base about?"
        print(f"Running query: '{query}'")
        
        response = await run_agent_graph(query, ai, db, project_id)
        
        print("\n--- RESPONSE ---")
        print(response)
        print("----------------")
        print("✅ LangGraph POC finished successfully.")
        
    except Exception as e:
        print(f"❌ Error during LangGraph execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_langgraph())
