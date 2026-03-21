"""
tests/test_agents.py - Unit tests for individual Healthcare Copilot agents.

Run with: pytest tests/ -v
Requires: OPENAI_API_KEY in environment
"""

import pytest
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.utils.models import ExtractedSymptom, PossibleCondition, RiskLevel


# ─── Symptom Extraction Agent Tests ──────────────────────────────────────────

class TestSymptomExtractionAgent:
    """Tests for SymptomExtractionAgent."""

    def test_fallback_extraction_common_symptoms(self):
        """Test that fallback extraction catches common symptoms."""
        from backend.agents.symptom_extraction_agent import SymptomExtractionAgent
        agent = SymptomExtractionAgent()
        
        symptoms = agent._fallback_extraction("I have fever and cough and headache")
        symptom_names = [s.name for s in symptoms]
        
        assert "fever" in symptom_names
        assert "cough" in symptom_names
        assert "headache" in symptom_names

    def test_fallback_extraction_empty_text(self):
        """Test fallback with no recognizable symptoms."""
        from backend.agents.symptom_extraction_agent import SymptomExtractionAgent
        agent = SymptomExtractionAgent()
        
        symptoms = agent._fallback_extraction("I feel generally unwell")
        assert isinstance(symptoms, list)

    def test_get_symptom_names(self):
        """Test helper to extract symptom names."""
        from backend.agents.symptom_extraction_agent import SymptomExtractionAgent
        agent = SymptomExtractionAgent()
        
        symptoms = [
            ExtractedSymptom(name="fever", severity="moderate"),
            ExtractedSymptom(name="cough"),
        ]
        names = agent.get_symptom_names(symptoms)
        assert names == ["fever", "cough"]


# ─── Risk Assessment Agent Tests ─────────────────────────────────────────────

class TestRiskAssessmentAgent:
    """Tests for rule-based risk assessment logic."""

    def test_high_risk_chest_pain(self):
        """Chest pain must always trigger HIGH risk."""
        from backend.agents.risk_assessment_agent import RiskAssessmentAgent
        agent = RiskAssessmentAgent()
        
        symptoms = [ExtractedSymptom(name="chest pain", severity="moderate")]
        conditions = []
        risk = agent._rule_based_risk(symptoms, conditions)
        assert risk == RiskLevel.HIGH

    def test_high_risk_shortness_of_breath(self):
        """Shortness of breath must trigger HIGH risk."""
        from backend.agents.risk_assessment_agent import RiskAssessmentAgent
        agent = RiskAssessmentAgent()
        
        symptoms = [ExtractedSymptom(name="shortness of breath")]
        risk = agent._rule_based_risk(symptoms, [])
        assert risk == RiskLevel.HIGH

    def test_low_risk_minor_cold(self):
        """Common cold symptoms should be LOW risk."""
        from backend.agents.risk_assessment_agent import RiskAssessmentAgent
        agent = RiskAssessmentAgent()
        
        symptoms = [
            ExtractedSymptom(name="runny nose", severity="mild"),
            ExtractedSymptom(name="sneezing"),
        ]
        conditions = [PossibleCondition(name="Common Cold", confidence="high", matching_symptoms=[])]
        risk = agent._rule_based_risk(symptoms, conditions)
        assert risk == RiskLevel.LOW

    def test_high_risk_stroke_symptoms(self):
        """Stroke-related symptoms must trigger HIGH risk."""
        from backend.agents.risk_assessment_agent import RiskAssessmentAgent
        agent = RiskAssessmentAgent()
        
        symptoms = [
            ExtractedSymptom(name="face drooping"),
            ExtractedSymptom(name="arm weakness"),
        ]
        risk = agent._rule_based_risk(symptoms, [])
        assert risk == RiskLevel.HIGH

    def test_fallback_rationale_high(self):
        """Fallback rationale for HIGH risk should mention emergency."""
        from backend.agents.risk_assessment_agent import RiskAssessmentAgent
        agent = RiskAssessmentAgent()
        
        symptoms = [ExtractedSymptom(name="chest pain")]
        rationale = agent._generate_fallback_rationale(RiskLevel.HIGH, symptoms)
        assert "emergency" in rationale.lower() or "immediate" in rationale.lower()

    def test_fallback_rationale_low(self):
        """Fallback rationale for LOW risk should mention self-care."""
        from backend.agents.risk_assessment_agent import RiskAssessmentAgent
        agent = RiskAssessmentAgent()
        
        symptoms = [ExtractedSymptom(name="mild headache")]
        rationale = agent._generate_fallback_rationale(RiskLevel.LOW, symptoms)
        assert "self-care" in rationale.lower() or "mild" in rationale.lower()


# ─── Retrieval Agent Tests ────────────────────────────────────────────────────

class TestRetrievalAgent:
    """Tests for RetrievalAgent context formatting."""

    def test_format_empty_snippets(self):
        """Empty snippets should return a no-context message."""
        from backend.agents.retrieval_agent import RetrievalAgent
        agent = RetrievalAgent()
        
        result = agent.format_context_for_prompt([])
        assert "No specific" in result

    def test_format_snippets(self):
        """Non-empty snippets should be numbered."""
        from backend.agents.retrieval_agent import RetrievalAgent
        agent = RetrievalAgent()
        
        snippets = ["Context A about fever", "Context B about cough"]
        result = agent.format_context_for_prompt(snippets)
        assert "[Context 1]" in result
        assert "[Context 2]" in result


# ─── Data Model Tests ────────────────────────────────────────────────────────

class TestDataModels:
    """Tests for Pydantic data model validation."""

    def test_extracted_symptom_defaults(self):
        """ExtractedSymptom should work with just a name."""
        s = ExtractedSymptom(name="headache")
        assert s.name == "headache"
        assert s.severity is None
        assert s.modifiers == []

    def test_possible_condition_creation(self):
        """PossibleCondition should validate correctly."""
        c = PossibleCondition(
            name="Influenza",
            confidence="high",
            matching_symptoms=["fever", "body aches"],
        )
        assert c.name == "Influenza"
        assert c.confidence == "high"
        assert len(c.matching_symptoms) == 2

    def test_risk_level_enum_values(self):
        """RiskLevel enum values should be correct."""
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MODERATE.value == "moderate"
        assert RiskLevel.HIGH.value == "high"


# ─── Integration-style (no LLM) Tests ────────────────────────────────────────

class TestOrchestratorIntegration:
    """Integration tests that don't require LLM (mock)."""

    def test_models_import_correctly(self):
        """All models should be importable."""
        from backend.utils.models import (
            ChatRequest, HealthcareResponse, ChatMessage,
            ExtractedSymptom, PossibleCondition, RiskLevel,
            PDFUploadResponse, QueryHistoryEntry
        )

    def test_config_imports(self):
        """Config module should import without errors."""
        from backend.config import (
            OPENAI_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
            MAX_CONVERSATION_HISTORY, MEDICAL_DISCLAIMER
        )
        assert CHUNK_SIZE > 0
        assert CHUNK_OVERLAP >= 0
        assert MAX_CONVERSATION_HISTORY > 0
        assert len(MEDICAL_DISCLAIMER) > 10


# ─── Run directly ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
