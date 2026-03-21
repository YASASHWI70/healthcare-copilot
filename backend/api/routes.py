"""
api/routes.py - FastAPI route handlers for the Healthcare Copilot.

Endpoints:
  POST /chat         - Main chat interaction endpoint
  POST /upload-pdf   - PDF text extraction endpoint
  GET  /health       - Health check
  GET  /history      - Query history (in-memory log)
"""

import uuid
import json
from datetime import datetime
from typing import List
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse

from backend.agents.orchestrator import HealthcareOrchestrator
from backend.utils.models import (
    ChatRequest, HealthcareResponse, PDFUploadResponse,
    QueryHistoryEntry, HealthStatus
)
from backend.config import OPENAI_MODEL
from backend.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# ─── Singletons ───────────────────────────────────────────────────────────────
# Orchestrator is initialized once and reused across requests
_orchestrator: HealthcareOrchestrator = None

# In-memory query history (replace with DB for production)
_query_history: List[QueryHistoryEntry] = []


def get_orchestrator() -> HealthcareOrchestrator:
    """Lazy initialization of the orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        logger.info("Initializing orchestrator on first request...")
        _orchestrator = HealthcareOrchestrator()
    return _orchestrator


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthStatus, tags=["System"])
async def health_check():
    """
    API health check endpoint.
    Returns system status and configuration info.
    """
    from backend.rag.vector_store import _vector_store
    return HealthStatus(
        status="ok",
        version="1.0.0",
        rag_index_loaded=_vector_store is not None,
        model=OPENAI_MODEL,
    )


@router.post("/chat", response_model=HealthcareResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Main chat endpoint — processes user symptoms through the multi-agent pipeline.

    The pipeline:
      1. Extracts structured symptoms from free text
      2. Retrieves relevant medical knowledge (RAG/FAISS)
      3. Performs clinical reasoning
      4. Assesses risk level
      5. Generates recommendations and explanation
      6. Returns structured response

    Args:
        request: ChatRequest with session_id, message, history, optional PDF text

    Returns:
        HealthcareResponse with full structured analysis
    """
    if not request.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty"
        )

    logger.info(f"Chat request: session={request.session_id}, msg_len={len(request.message)}")

    try:
        orchestrator = get_orchestrator()
        response = orchestrator.run(
            session_id=request.session_id,
            user_message=request.message,
            conversation_history=request.conversation_history,
            pdf_text=request.pdf_text,
        )

        # Log to history
        _query_history.append(QueryHistoryEntry(
            session_id=request.session_id,
            timestamp=datetime.now().isoformat(),
            query=request.message[:100],
            risk_level=response.risk_level if isinstance(response.risk_level, str) else response.risk_level.value,
            conditions_count=len(response.possible_conditions),
            symptoms_count=len(response.extracted_symptoms),
        ))

        # Keep last 100 entries
        if len(_query_history) > 100:
            _query_history.pop(0)

        return response

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {str(e)}"
        )


@router.post("/upload-pdf", response_model=PDFUploadResponse, tags=["PDF"])
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and extract text from a medical PDF report.

    Accepts a PDF file and returns the extracted text content
    which can then be included in subsequent chat requests.

    Args:
        file: Uploaded PDF file

    Returns:
        PDFUploadResponse with extracted text and metadata
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported"
        )

    # File size check (max 10MB)
    MAX_SIZE = 10 * 1024 * 1024
    content = await file.read()
    if len(content) > MAX_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds 10MB limit"
        )

    try:
        from pypdf import PdfReader
        import io

        reader = PdfReader(io.BytesIO(content))
        page_count = len(reader.pages)

        extracted_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                extracted_text += page_text + "\n\n"

        extracted_text = extracted_text.strip()

        if not extracted_text:
            return PDFUploadResponse(
                filename=file.filename,
                extracted_text="",
                page_count=page_count,
                success=False,
                error="No extractable text found in PDF (may be a scanned image)"
            )

        logger.info(f"PDF extracted: {file.filename}, {page_count} pages, {len(extracted_text)} chars")

        return PDFUploadResponse(
            filename=file.filename,
            extracted_text=extracted_text[:5000],  # Limit to 5000 chars
            page_count=page_count,
            success=True,
        )

    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"PDF processing failed: {str(e)}"
        )


@router.get("/history", response_model=List[QueryHistoryEntry], tags=["History"])
async def get_history(limit: int = 20):
    """
    Retrieve recent query history.

    Args:
        limit: Maximum number of entries to return (default 20, max 100)

    Returns:
        List of recent QueryHistoryEntry objects
    """
    limit = min(limit, 100)
    return list(reversed(_query_history))[:limit]


@router.delete("/history", tags=["History"])
async def clear_history():
    """Clear the query history."""
    _query_history.clear()
    return {"message": "History cleared"}


@router.post("/rebuild-index", tags=["System"])
async def rebuild_rag_index():
    """
    Force rebuild the FAISS index from knowledge documents.
    Useful after adding new knowledge files.
    """
    try:
        from backend.rag.vector_store import build_vector_store, _vector_store
        import backend.rag.vector_store as vs
        vs._vector_store = build_vector_store(force_rebuild=True)
        return {"message": "FAISS index rebuilt successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Index rebuild failed: {str(e)}"
        )
