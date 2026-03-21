"""
main.py - FastAPI application entrypoint for Healthcare Copilot.

Starts the backend server with:
  - CORS configuration for Streamlit frontend
  - Lifespan events for RAG index initialization
  - API router registration
  - OpenAPI documentation
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from backend.api.routes import router
from backend.config import BACKEND_HOST, BACKEND_PORT, OPENAI_API_KEY
from backend.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager: runs startup and shutdown logic.
    Pre-loads the FAISS vector index on startup to avoid
    cold-start latency on the first request.
    """
    # ─── Startup ─────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  Healthcare Copilot Backend Starting...")
    logger.info("=" * 60)

    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not set! Please configure your .env file.")
    else:
        logger.info(f"OpenAI API key configured ({'*' * 8}{OPENAI_API_KEY[-4:]})")

    # Pre-load FAISS index
    try:
        logger.info("Pre-loading FAISS vector index...")
        from backend.rag.vector_store import get_vector_store
        get_vector_store()
        logger.info("FAISS index loaded successfully ✓")
    except Exception as e:
        logger.warning(f"FAISS index pre-load failed (will retry on first request): {e}")

    logger.info("Backend ready to accept requests ✓")
    logger.info("=" * 60)

    yield  # App runs here

    # ─── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("Healthcare Copilot Backend shutting down...")


# ─── App Instance ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Healthcare Copilot API",
    description=(
        "Multi-Agent Generative AI Healthcare Copilot — "
        "Clinical Decision Support using LLMs, RAG, and Agent Orchestration.\n\n"
        "⚠️ **This is NOT medical advice. For educational and informational purposes only.**"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",   # Streamlit default
        "http://127.0.0.1:8501",
        "http://localhost:3000",   # React dev server
        "*",                       # Allow all for development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routes ───────────────────────────────────────────────────────────────────
app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with basic info."""
    return {
        "service": "Healthcare Copilot API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "disclaimer": "This is not medical advice."
    }


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=BACKEND_HOST,
        port=BACKEND_PORT,
        reload=True,  # Disable in production
        log_level="info",
    )
