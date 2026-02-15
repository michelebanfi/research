import os
import json
import hashlib
import time
from typing import Any, Optional, Dict
from pathlib import Path
from src.config import Config

class Cache:
    """
    REQ-PERF-01: Disk-based cache with LRU eviction for expensive operations.
    """
    
    def __init__(self, cache_dir: str = ".cache", max_size: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = Config.CACHE_ENABLED
        self.ttl = Config.CACHE_TTL  # Seconds
        self.max_size = max_size
        self._access_file = self.cache_dir / "_access_stats.json"
        
        # Load access stats or initialize
        self._access_stats = {}
        self._load_stats()
        
    def _load_stats(self):
        if self._access_file.exists():
            try:
                with open(self._access_file, "r") as f:
                    self._access_stats = json.load(f)
            except Exception:
                self._access_stats = {}

    def _save_stats(self):
        try:
            with open(self._access_file, "w") as f:
                json.dump(self._access_stats, f)
        except Exception:
            pass

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
                # Lazy expire
                self._delete(key)
                return None
            
            # Update access time
            self._access_stats[key] = time.time()
            self._save_stats()
                
            return entry["value"]
        except Exception:
            return None
            
    def set(self, prefix: str, key_data: Any, value: Any):
        """Store item in cache with LRU eviction."""
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
            
            self._access_stats[key] = time.time()
            self._save_stats()
            
            # Check size and evict if needed
            self._evict_if_needed()
            
        except Exception as e:
            print(f"Cache write error: {e}")

    def _delete(self, key: str):
        path = self._get_path(key)
        if path.exists():
            path.unlink()
        if key in self._access_stats:
            del self._access_stats[key]

    def _evict_if_needed(self):
        """Evict least recently used items until under max_size."""
        # Clean up stats for missing files first (sync)
        existing_keys = [f.stem for f in self.cache_dir.glob("*.json") if f.name != "_access_stats.json"]
        self._access_stats = {k: v for k, v in self._access_stats.items() if k in existing_keys}
        
        if len(existing_keys) <= self.max_size:
            return
            
        # Sort by access time (oldest first)
        # Verify keys exist in stats, otherwise use 0
        sorted_keys = sorted(existing_keys, key=lambda k: self._access_stats.get(k, 0))
        
        # Evict
        num_to_delete = len(existing_keys) - self.max_size
        for i in range(num_to_delete):
            self._delete(sorted_keys[i])
            
        self._save_stats()

    def clear(self):
        """Clear all cache."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir()

