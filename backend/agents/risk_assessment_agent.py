"""
agents/risk_assessment_agent.py - Assigns risk levels using rule-based + LLM logic.

Risk levels:
  - LOW: Non-urgent, self-care appropriate
  - MODERATE: Seek medical attention within 24 hours
  - HIGH: Immediate medical attention / emergency services

Rule-based approach is used first (deterministic), with LLM
providing nuanced rationale and handling edge cases.
"""

from typing import List, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from backend.config import OPENAI_API_KEY, OPENAI_MODEL
from backend.utils.models import ExtractedSymptom, PossibleCondition, RiskLevel
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# ─── Rule-Based Risk Criteria ─────────────────────────────────────────────────

# Symptoms that immediately trigger HIGH risk
HIGH_RISK_SYMPTOMS = {
    "chest pain", "chest pressure", "chest tightness",
    "shortness of breath", "difficulty breathing", "breathlessness",
    "sudden severe headache", "thunderclap headache", "worst headache of life",
    "face drooping", "arm weakness", "speech difficulty", "slurred speech",
    "loss of consciousness", "unconscious", "unresponsive",
    "severe allergic reaction", "anaphylaxis", "throat swelling",
    "stroke", "heart attack",
    "suicidal thoughts", "suicidal ideation",
    "seizure", "convulsion",
    "coughing blood", "vomiting blood", "blood in stool",
    "sudden vision loss", "sudden weakness",
    "severe abdominal pain", "rigid abdomen",
    "blue lips", "cyanosis",
}

# Symptom combinations that escalate to MODERATE
MODERATE_RISK_SYMPTOMS = {
    "high fever", "persistent fever", "fever over 39",
    "severe pain", "debilitating pain",
    "persistent vomiting", "unable to keep fluids down",
    "signs of dehydration", "severe diarrhea",
    "confusion", "altered mental status",
    "palpitations", "rapid heartbeat",
    "severe fatigue", "extreme fatigue",
}

# Conditions that are inherently HIGH risk
HIGH_RISK_CONDITIONS = {
    "stroke", "heart attack", "myocardial infarction", "acute coronary syndrome",
    "pulmonary embolism", "anaphylaxis", "sepsis", "meningitis",
    "appendicitis", "pneumothorax", "aortic dissection",
    "suicidal ideation", "overdose",
}

MODERATE_RISK_CONDITIONS = {
    "pneumonia", "asthma exacerbation", "pyelonephritis",
    "heart failure", "hypertensive crisis", "severe dehydration",
    "diabetic ketoacidosis",
}


class RiskAssessmentAgent:
    """
    Hybrid risk assessment using rules + LLM reasoning.

    Rule-based logic provides fast, deterministic safety checks.
    LLM provides nuanced rationale and handles edge cases.
    """

    SYSTEM_PROMPT = """You are a clinical risk assessment AI. Your role is to evaluate 
the overall risk level of a patient's presentation and provide a clear, 
non-alarmist but honest risk rationale.

Risk levels:
- LOW: Can likely manage with self-care; see a GP if it doesn't improve
- MODERATE: Should see a doctor within 24 hours; urgent care may be appropriate
- HIGH: Requires immediate medical attention; may need emergency services

CRITICAL: If ANY red flag symptoms are present (chest pain, stroke symptoms, 
difficulty breathing, loss of consciousness), ALWAYS assign HIGH risk."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0.0,
            openai_api_key=OPENAI_API_KEY,
        )
        logger.info("RiskAssessmentAgent initialized")

    def _rule_based_risk(
        self,
        symptoms: List[ExtractedSymptom],
        conditions: List[PossibleCondition],
    ) -> RiskLevel:
        """
        Fast, deterministic rule-based risk check.

        Args:
            symptoms: Extracted symptoms
            conditions: Possible conditions

        Returns:
            Risk level or None if rules don't apply
        """
        symptom_names = {s.name.lower() for s in symptoms}
        condition_names = {c.name.lower() for c in conditions}

        # Check HIGH risk symptoms
        for symptom in symptom_names:
            for high_risk in HIGH_RISK_SYMPTOMS:
                if high_risk in symptom or symptom in high_risk:
                    logger.info(f"HIGH risk triggered by symptom: {symptom}")
                    return RiskLevel.HIGH

        # Check severe symptoms
        for symptom in symptoms:
            if symptom.severity == "severe":
                for moderate in MODERATE_RISK_SYMPTOMS:
                    if moderate in symptom.name.lower():
                        return RiskLevel.HIGH  # Severe + moderate trigger = HIGH

        # Check HIGH risk conditions
        for condition in condition_names:
            for high_risk in HIGH_RISK_CONDITIONS:
                if high_risk in condition:
                    logger.info(f"HIGH risk triggered by condition: {condition}")
                    return RiskLevel.HIGH

        # Check MODERATE risk symptoms
        for symptom in symptom_names:
            for moderate in MODERATE_RISK_SYMPTOMS:
                if moderate in symptom or symptom in moderate:
                    return RiskLevel.MODERATE

        # Check MODERATE risk conditions
        for condition in condition_names:
            for moderate in MODERATE_RISK_CONDITIONS:
                if moderate in condition:
                    return RiskLevel.MODERATE

        # Check for high-confidence severe conditions
        for condition in conditions:
            if condition.confidence == "high" and condition.name.lower() in MODERATE_RISK_CONDITIONS:
                return RiskLevel.MODERATE

        return RiskLevel.LOW

    def assess(
        self,
        symptoms: List[ExtractedSymptom],
        conditions: List[PossibleCondition],
        rag_context: str,
    ) -> Tuple[RiskLevel, str]:
        """
        Assess overall risk level with explanation.

        Args:
            symptoms: Extracted symptoms
            conditions: Possible conditions from reasoning agent
            rag_context: Medical knowledge context

        Returns:
            Tuple of (risk_level, rationale_string)
        """
        # Rule-based check first (deterministic)
        rule_risk = self._rule_based_risk(symptoms, conditions)

        # Build prompt for LLM rationale
        symptom_str = ", ".join(
            f"{s.name}" + (f" ({s.severity})" if s.severity else "")
            for s in symptoms
        ) or "No specific symptoms extracted"

        condition_str = ", ".join(
            f"{c.name} ({c.confidence} confidence)" for c in conditions
        ) or "No conditions identified"

        prompt = f"""CLINICAL ASSESSMENT:

Symptoms: {symptom_str}
Possible Conditions: {condition_str}
Rule-Based Risk Level: {rule_risk.value.upper()}

MEDICAL CONTEXT:
{rag_context[:800]}

---

Based on this clinical picture:
1. Confirm or override the rule-based risk level (LOW / MODERATE / HIGH)
2. Provide a clear 2-3 sentence rationale explaining:
   - Why this risk level was assigned
   - What specific symptoms or conditions drove this assessment
   - What the patient should do

Format your response as:
RISK_LEVEL: [LOW|MODERATE|HIGH]
RATIONALE: [Your explanation]"""

        try:
            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]

            response = self.llm.invoke(messages)
            content = response.content.strip()

            # Parse response
            final_risk = rule_risk
            rationale = ""

            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("RISK_LEVEL:"):
                    level_str = line.replace("RISK_LEVEL:", "").strip().lower()
                    # Never downgrade from HIGH (safety)
                    if rule_risk == RiskLevel.HIGH:
                        final_risk = RiskLevel.HIGH
                    elif level_str in ("low", "moderate", "high"):
                        final_risk = RiskLevel(level_str)
                elif line.startswith("RATIONALE:"):
                    rationale = line.replace("RATIONALE:", "").strip()

            # If rationale wasn't parsed, use the full response
            if not rationale:
                rationale = content

            logger.info(f"Risk assessment: {final_risk.value} (rules said: {rule_risk.value})")
            return final_risk, rationale

        except Exception as e:
            logger.error(f"RiskAssessmentAgent error: {e}")
            # Fall back to rule-based
            fallback_rationale = self._generate_fallback_rationale(rule_risk, symptoms)
            return rule_risk, fallback_rationale

    def _generate_fallback_rationale(
        self, risk: RiskLevel, symptoms: List[ExtractedSymptom]
    ) -> str:
        """Generate a basic rationale when LLM fails."""
        symptom_names = [s.name for s in symptoms[:3]]

        if risk == RiskLevel.HIGH:
            return (
                f"HIGH RISK: One or more symptoms ({', '.join(symptom_names)}) "
                "suggest a potentially serious condition requiring immediate medical attention. "
                "Please seek emergency care or call emergency services."
            )
        elif risk == RiskLevel.MODERATE:
            return (
                f"MODERATE RISK: The symptoms ({', '.join(symptom_names)}) "
                "warrant medical evaluation within the next 24 hours. "
                "Please contact your doctor or visit an urgent care center."
            )
        else:
            return (
                f"LOW RISK: The symptoms ({', '.join(symptom_names)}) appear mild "
                "and can likely be managed with self-care. Monitor for any worsening "
                "and consult a doctor if symptoms persist beyond 2-3 days."
            )
