print("Start import debug")
try:
    import sys
    print("sys imported")
    from src.ingestion import DoclingParser
    print("DoclingParser imported")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Exception: {e}")
print("End import debug")
