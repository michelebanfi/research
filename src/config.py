import os
from dotenv import load_dotenv

# Load env vars from .env if present
load_dotenv()

class Config:
    SUPABASE_URL = os.environ.get("SUPABASE_URL")
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
    OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen:2.5")
    OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    @classmethod
    def validate(cls):
        """Ensures critical environment variables are set."""
        if not cls.SUPABASE_URL or not cls.SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
