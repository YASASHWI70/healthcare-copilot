"""
config.py - Centralized configuration management for Healthcare Copilot
Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# ─── Project Paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_DIR = DATA_DIR / "medical_knowledge"
FAISS_INDEX_PATH = str(DATA_DIR / "faiss_index")
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ─── OpenAI / LLM ────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# ─── RAG ─────────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))

# ─── Conversation Memory ──────────────────────────────────────────────────────
MAX_CONVERSATION_HISTORY: int = int(os.getenv("MAX_CONVERSATION_HISTORY", "10"))

# ─── Backend ─────────────────────────────────────────────────────────────────
BACKEND_HOST: str = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT: int = int(os.getenv("BACKEND_PORT", "8000"))

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE: str = os.getenv("LOG_FILE", str(LOG_DIR / "copilot.log"))

# ─── Disclaimer ──────────────────────────────────────────────────────────────
MEDICAL_DISCLAIMER = (
    "⚠️ DISCLAIMER: This tool is for informational and educational purposes only. "
    "It does NOT constitute medical advice, diagnosis, or treatment. "
    "Always consult a qualified healthcare professional for medical concerns. "
    "In an emergency, call your local emergency services (911/112/999) immediately."
)
