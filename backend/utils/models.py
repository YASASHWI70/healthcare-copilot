"""
utils/models.py - Pydantic data models for the Healthcare Copilot system.
Defines the structured input/output contracts for all API endpoints and agents.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    UNKNOWN = "unknown"


class AgentStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"


# ─── Chat Input/Output ────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    """Single message in a conversation"""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Incoming chat request from the user"""
    session_id: str = Field(..., description="Unique session identifier")
    message: str = Field(..., description="User's symptom description or follow-up")
    conversation_history: List[ChatMessage] = Field(
        default_factory=list,
        description="Previous messages for context"
    )
    pdf_text: Optional[str] = Field(
        None,
        description="Extracted text from uploaded medical PDF"
    )


class ExtractedSymptom(BaseModel):
    """A single structured symptom extracted from user text"""
    name: str = Field(..., description="Symptom name (e.g., 'fever')")
    severity: Optional[str] = Field(None, description="mild / moderate / severe")
    duration: Optional[str] = Field(None, description="e.g., '2 days', 'since morning'")
    location: Optional[str] = Field(None, description="Body location if applicable")
    modifiers: Optional[List[str]] = Field(
        default_factory=list,
        description="Qualifiers like 'intermittent', 'worsening', 'with exertion'"
    )


class PossibleCondition(BaseModel):
    """A potential medical condition with supporting evidence"""
    name: str = Field(..., description="Condition name")
    confidence: str = Field(..., description="high / moderate / low match to symptoms")
    matching_symptoms: List[str] = Field(
        default_factory=list,
        description="Which user symptoms support this condition"
    )
    key_differentiators: Optional[str] = Field(
        None,
        description="What makes this condition more/less likely"
    )


class ReasoningStep(BaseModel):
    """One step in the clinical reasoning chain"""
    step_number: int
    title: str
    content: str


class HealthcareResponse(BaseModel):
    """
    Structured response from the multi-agent pipeline.
    This is the main output format returned to the frontend.
    """
    session_id: str
    original_query: str

    # Agent outputs
    extracted_symptoms: List[ExtractedSymptom] = Field(default_factory=list)
    possible_conditions: List[PossibleCondition] = Field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.UNKNOWN
    risk_rationale: str = ""
    reasoning_steps: List[ReasoningStep] = Field(default_factory=list)
    explanation: str = Field(..., description="Plain-language explanation for the patient")
    recommendations: List[str] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)
    rag_context_used: List[str] = Field(
        default_factory=list,
        description="Knowledge snippets retrieved from vector DB"
    )
    assistant_message: str = Field(
        ...,
        description="Conversational reply to display in chat"
    )
    disclaimer: str = ""
    agent_trace: Optional[Dict[str, AgentStatus]] = Field(
        default_factory=dict,
        description="Execution status of each agent"
    )

    class Config:
        use_enum_values = True


# ─── PDF Upload ───────────────────────────────────────────────────────────────

class PDFUploadResponse(BaseModel):
    """Response after PDF text extraction"""
    filename: str
    extracted_text: str
    page_count: int
    success: bool
    error: Optional[str] = None


# ─── History ─────────────────────────────────────────────────────────────────

class QueryHistoryEntry(BaseModel):
    """Log entry for a single query"""
    session_id: str
    timestamp: str
    query: str
    risk_level: str
    conditions_count: int
    symptoms_count: int


class HealthStatus(BaseModel):
    """API health check response"""
    status: str = "ok"
    version: str = "1.0.0"
    rag_index_loaded: bool = False
    model: str = ""
