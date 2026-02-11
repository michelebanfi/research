import sys
import os

print("Starting import test...")
try:
    print("Importing langchain_openai...")
    from langchain_openai import ChatOpenAI
    print("✅ langchain_openai imported")
except ImportError as e:
    print(f"❌ langchain_openai failed: {e}")

try:
    print("Importing langgraph...")
    from langgraph.graph import StateGraph
    print("✅ langgraph imported")
except ImportError as e:
    print(f"❌ langgraph failed: {e}")

print("Import test detailed finished.")
