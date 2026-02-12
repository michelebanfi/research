import sys
import os
import asyncio
import nest_asyncio

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Apply nest_asyncio
nest_asyncio.apply()

from src.ai_engine import AIEngine
from src.reasoning_agent import ReasoningAgent
from dotenv import load_dotenv

load_dotenv()

async def test_reasoning_model_name():
    print("Testing ReasoningAgent model_name capture...")
    try:
        ai = AIEngine()
        agent = ReasoningAgent(ai)
        
        # Simple task that requires no context
        task = "Calculate 2 + 2 in Python"
        print(f"Task: {task}")
        
        response = await agent.run(task)
        
        print("\n--- RESPONSE ---")
        print(f"Success: {response.success}")
        print(f"Model Name: '{response.model_name}'")
        
        if response.model_name and response.model_name != "unknown":
            print("✅ Model name captured successfully.")
        else:
            print("❌ Model name is missing or 'unknown'.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_reasoning_model_name())
