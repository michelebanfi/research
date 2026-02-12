import os
import json
import hashlib
import time
from typing import Any, Optional, Dict
from pathlib import Path
from src.config import Config

class Cache:
    """
    REQ-PERF-01: Simple disk-based cache for expensive operations (Embeddings, LLM calls).
    """
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = Config.CACHE_ENABLED
        self.ttl = Config.CACHE_TTL  # Seconds
        
    def _get_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"
        
    def _generate_key(self, prefix: str, data: Any) -> str:
        """Generates a stable cache key."""
        if isinstance(data, dict) or isinstance(data, list):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
            
        return hashlib.md5(f"{prefix}:{data_str}".encode()).hexdigest()
        
    def get(self, prefix: str, key_data: Any) -> Optional[Any]:
        """Retrieve item from cache if valid."""
        if not self.enabled:
            return None
            
        key = self._generate_key(prefix, key_data)
        path = self._get_path(key)
        
        if not path.exists():
            return None
            
        try:
            with open(path, "r") as f:
                entry = json.load(f)
                
            # Check TTL
            if time.time() - entry["timestamp"] > self.ttl:
                return None
                
            return entry["value"]
        except Exception:
            return None
            
    def set(self, prefix: str, key_data: Any, value: Any):
        """Store item in cache."""
        if not self.enabled:
            return
            
        key = self._generate_key(prefix, key_data)
        path = self._get_path(key)
        
        try:
            entry = {
                "timestamp": time.time(),
                "value": value
            }
            with open(path, "w") as f:
                json.dump(entry, f)
        except Exception as e:
            print(f"Cache write error: {e}")

    def clear(self):
        """Clear all cache."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir()
