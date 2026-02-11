import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_imports():
    print("Testing imports...")
    try:
        from src.ui import chat
        print("✅ src.ui.chat imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import src.ui.chat: {e}")
        return

    try:
        from src.ui import ingest
        print("✅ src.ui.ingest imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import src.ui.ingest: {e}")
        return

    try:
        from src.ui import graph
        print("✅ src.ui.graph imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import src.ui.graph: {e}")
        return

    print("All UI modules imported successfully.")

if __name__ == "__main__":
    test_imports()
