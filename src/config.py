import os
from dotenv import load_dotenv

# Load env vars from .env if present
load_dotenv()

class Config:
    SUPABASE_URL = os.environ.get("SUPABASE_URL")
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
    OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openrouter/free")
    OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY", "")
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    EMBEDDING_DIMENSION = int(os.environ.get("EMBEDDING_DIMENSION", "1024"))  # Your nomic-embed-text = 1024
    # REQ-STABILITY-01: Context window limit for LLM
    MODEL_CONTEXT_LIMIT = int(os.environ.get("MODEL_CONTEXT_LIMIT", "4096"))
    
    # REQ-IMP-05: Sandbox execution timeout
    SANDBOX_TIMEOUT_S = float(os.environ.get("SANDBOX_TIMEOUT_S", "5.0"))
    
    # REQ-IMP-06: API rate limiting
    API_MAX_CONCURRENT_CALLS = int(os.environ.get("API_MAX_CONCURRENT_CALLS", "5"))
    
    # REQ-IMP-07: Retry logic
    API_RETRY_ATTEMPTS = int(os.environ.get("API_RETRY_ATTEMPTS", "3"))
    API_RETRY_MIN_WAIT_S = float(os.environ.get("API_RETRY_MIN_WAIT_S", "1.0"))
    API_RETRY_MAX_WAIT_S = float(os.environ.get("API_RETRY_MAX_WAIT_S", "10.0"))
    
    # REQ-IMP-08: Embedding cache size
    EMBEDDING_CACHE_SIZE = int(os.environ.get("EMBEDDING_CACHE_SIZE", "1000"))

    # RERANK-01: Re-ranking model name
    # Options: ms-marco-MiniLM-L-12-v2 (default), rank-T5-flan, rank_zephyr_7b_v1_full
    RERANK_MODEL_NAME = os.environ.get("RERANK_MODEL_NAME", "ms-marco-MiniLM-L-12-v2")

    @classmethod
    def validate(cls):
        """Ensures critical environment variables are set."""
        if not cls.SUPABASE_URL or not cls.SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
