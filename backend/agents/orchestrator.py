"""
agents/orchestrator.py - Central orchestrator managing the multi-agent pipeline.

The Orchestrator coordinates all agents in sequence:
  1. ConversationAgent    → context management
  2. SymptomExtractionAgent → extract structured symptoms
  3. RetrievalAgent       → fetch RAG context from FAISS
  4. ReasoningAgent       → differential diagnosis + reasoning steps
  5. RiskAssessmentAgent  → risk level + rationale
  6. RecommendationAgent  → next steps + plain explanation
  7. ConversationAgent    → final conversational reply

All errors are handled gracefully so the pipeline degrades
gracefully rather than failing completely.
"""

import time
from typing import List, Optional

# from backend.agents.conversation_agent import ConversationAgent
# from backend.agents.symptom_extraction_agent import SymptomExtractionAgent
# from backend.agents.retrieval_agent import RetrievalAgent
# from backend.agents.reasoning_agent import ReasoningAgent
# from backend.agents.risk_assessment_agent import RiskAssessmentAgent
# from backend.agents.recommendation_agent import RecommendationAgent
# from backend.utils.models import (
#     ChatMessage, HealthcareResponse, RiskLevel, AgentStatus
# )
from backend.config import MEDICAL_DISCLAIMER
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class HealthcareOrchestrator:
    """
    Master orchestrator for the Healthcare Copilot multi-agent system.

    Manages agent instantiation, execution order, error handling,
    and result aggregation into a structured HealthcareResponse.
    """

    def __init__(self):
        logger.info("Initializing HealthcareOrchestrator and all agents...")

        self.conversation_agent = ConversationAgent()
        self.symptom_agent = SymptomExtractionAgent()
        self.retrieval_agent = RetrievalAgent()
        self.reasoning_agent = ReasoningAgent()
        self.risk_agent = RiskAssessmentAgent()
        self.recommendation_agent = RecommendationAgent()

        logger.info("All agents initialized successfully")

    def run(
        self,
        session_id: str,
        user_message: str,
        conversation_history: List[ChatMessage],
        pdf_text: Optional[str] = None,
    ) -> HealthcareResponse:
        """
        Execute the full multi-agent pipeline for a healthcare query.

        Args:
            session_id: Unique session identifier
            user_message: User's symptom description
            conversation_history: Prior conversation turns
            pdf_text: Optional extracted text from medical PDF

        Returns:
            Structured HealthcareResponse with all agent outputs
        """
        start_time = time.time()
        agent_trace = {}

        logger.info(f"[Session {session_id}] Pipeline started: '{user_message[:50]}...'")

        # ─── STEP 1: Symptom Extraction ───────────────────────────────────────
        try:
            symptoms = self.symptom_agent.extract(user_message, pdf_text)
            agent_trace["symptom_extraction"] = AgentStatus.SUCCESS
            logger.info(f"[Step 1] Extracted {len(symptoms)} symptoms")
        except Exception as e:
            logger.error(f"[Step 1] Symptom extraction failed: {e}")
            symptoms = []
            agent_trace["symptom_extraction"] = AgentStatus.ERROR

        # ─── STEP 2: RAG Retrieval ────────────────────────────────────────────
        try:
            rag_snippets = self.retrieval_agent.retrieve(
                symptoms, user_message, top_k=5
            )
            rag_context = self.retrieval_agent.format_context_for_prompt(rag_snippets)
            agent_trace["retrieval"] = AgentStatus.SUCCESS
            logger.info(f"[Step 2] Retrieved {len(rag_snippets)} context snippets")
        except Exception as e:
            logger.error(f"[Step 2] RAG retrieval failed: {e}")
            rag_snippets = []
            rag_context = ""
            agent_trace["retrieval"] = AgentStatus.ERROR

        # ─── STEP 3: Medical Reasoning ────────────────────────────────────────
        try:
            conditions, reasoning_steps = self.reasoning_agent.reason(
                symptoms, rag_context, user_message
            )
            agent_trace["reasoning"] = AgentStatus.SUCCESS
            logger.info(f"[Step 3] Identified {len(conditions)} conditions, {len(reasoning_steps)} reasoning steps")
        except Exception as e:
            logger.error(f"[Step 3] Reasoning failed: {e}")
            conditions = []
            reasoning_steps = []
            agent_trace["reasoning"] = AgentStatus.ERROR

        # ─── STEP 4: Risk Assessment ──────────────────────────────────────────
        try:
            risk_level, risk_rationale = self.risk_agent.assess(
                symptoms, conditions, rag_context
            )
            agent_trace["risk_assessment"] = AgentStatus.SUCCESS
            logger.info(f"[Step 4] Risk level: {risk_level.value}")
        except Exception as e:
            logger.error(f"[Step 4] Risk assessment failed: {e}")
            risk_level = RiskLevel.UNKNOWN
            risk_rationale = "Unable to assess risk. Please consult a healthcare professional."
            agent_trace["risk_assessment"] = AgentStatus.ERROR

        # ─── STEP 5: Recommendations ──────────────────────────────────────────
        try:
            recommendations = self.recommendation_agent.generate_recommendations(
                symptoms, conditions, risk_level, rag_context
            )
            agent_trace["recommendation"] = AgentStatus.SUCCESS
            logger.info(f"[Step 5] Generated {len(recommendations)} recommendations")
        except Exception as e:
            logger.error(f"[Step 5] Recommendation generation failed: {e}")
            recommendations = ["Please consult a healthcare professional for guidance."]
            agent_trace["recommendation"] = AgentStatus.ERROR

        # ─── STEP 6: Plain-Language Explanation ───────────────────────────────
        try:
            reasoning_summary = (
                reasoning_steps[1].content if len(reasoning_steps) > 1 else ""
            )
            explanation = self.recommendation_agent.generate_explanation(
                symptoms, conditions, risk_level, reasoning_summary
            )
            agent_trace["explanation"] = AgentStatus.SUCCESS
        except Exception as e:
            logger.error(f"[Step 6] Explanation generation failed: {e}")
            explanation = (
                "Based on your symptoms, a detailed analysis has been completed. "
                "Please review the findings and consult a healthcare professional."
            )
            agent_trace["explanation"] = AgentStatus.ERROR

        # ─── STEP 7: Follow-Up Questions ──────────────────────────────────────
        try:
            symptom_names = [s.name for s in symptoms]
            condition_names = [c.name for c in conditions]
            follow_up_questions = self.conversation_agent.generate_follow_up_questions(
                symptom_names, condition_names
            )
            agent_trace["follow_up"] = AgentStatus.SUCCESS
        except Exception as e:
            logger.error(f"[Step 7] Follow-up generation failed: {e}")
            follow_up_questions = []
            agent_trace["follow_up"] = AgentStatus.ERROR

        # ─── STEP 8: Conversational Reply ─────────────────────────────────────
        try:
            analysis_summary = (
                f"Risk Level: {risk_level.value.upper()}\n"
                f"Possible conditions: {', '.join(c.name for c in conditions[:2])}\n"
                f"Symptoms found: {', '.join(s.name for s in symptoms[:3])}\n"
                f"Key point: {risk_rationale[:150]}"
            )
            assistant_message = self.conversation_agent.generate_conversational_reply(
                user_message, conversation_history, analysis_summary
            )
            agent_trace["conversation"] = AgentStatus.SUCCESS
        except Exception as e:
            logger.error(f"[Step 8] Conversational reply failed: {e}")
            assistant_message = (
                "I've analyzed your symptoms. Please review the detailed report below. "
                "Remember, this is for informational purposes only."
            )
            agent_trace["conversation"] = AgentStatus.ERROR

        # ─── Assemble Final Response ──────────────────────────────────────────
        elapsed = round(time.time() - start_time, 2)
        logger.info(f"[Session {session_id}] Pipeline completed in {elapsed}s")

        return HealthcareResponse(
            session_id=session_id,
            original_query=user_message,
            extracted_symptoms=symptoms,
            possible_conditions=conditions,
            risk_level=risk_level,
            risk_rationale=risk_rationale,
            reasoning_steps=reasoning_steps,
            explanation=explanation,
            recommendations=recommendations,
            follow_up_questions=follow_up_questions,
            rag_context_used=[s[:200] for s in rag_snippets[:3]],  # Truncate for response
            assistant_message=assistant_message,
            disclaimer=MEDICAL_DISCLAIMER,
            agent_trace=agent_trace,
        )
