"""
agents/recommendation_agent.py - Generates actionable next steps and recommendations.

Produces:
  - Actionable next steps tailored to risk level
  - Symptom-specific precautions
  - Plain-language explanation
  - When to seek emergency care
"""

# from typing import List

# from langchain_openai import ChatOpenAI
# from langchain_core.messages import SystemMessage, HumanMessage

# from backend.config import OPENAI_API_KEY, OPENAI_MODEL
# from backend.utils.models import (
#     ExtractedSymptom, PossibleCondition, RiskLevel
# )
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class RecommendationAgent:
    """
    Generates personalized, risk-appropriate health recommendations.

    Combines rule-based safety recommendations with LLM-generated
    explanations and next steps tailored to the specific presentation.
    """

    SYSTEM_PROMPT = """You are a health education AI assistant. Generate clear, 
actionable health recommendations for patients based on their symptoms and risk level.

RULES:
- Be specific and actionable (not vague)
- Match urgency to risk level
- Include both self-care AND professional care guidance
- Use plain language — avoid medical jargon
- For HIGH risk: lead with "CALL EMERGENCY SERVICES" or "Go to ER NOW"
- Always include a disclaimer that this is NOT medical advice
- Maximum 6 recommendations, each 1-2 sentences

Return ONLY a numbered list of recommendations, no preamble."""

    # Baseline recommendations by risk level
    BASELINE_RECOMMENDATIONS = {
        RiskLevel.LOW: [
            "Rest and get adequate sleep to support your body's recovery.",
            "Stay well hydrated — drink 8-10 glasses of water daily.",
            "Monitor your symptoms and note any changes or new developments.",
            "Consider over-the-counter medications for symptom relief (follow package instructions).",
            "Schedule a GP appointment if symptoms persist beyond 3-5 days or worsen.",
        ],
        RiskLevel.MODERATE: [
            "⚠️ Contact your doctor or visit urgent care within the next 24 hours.",
            "Do NOT ignore worsening symptoms — seek care sooner if they escalate.",
            "Avoid strenuous activity until you have been evaluated.",
            "Keep a symptom diary (when symptoms started, severity, what helps/worsens).",
            "If you develop chest pain, difficulty breathing, or confusion — go to the ER immediately.",
        ],
        RiskLevel.HIGH: [
            "🚨 SEEK IMMEDIATE MEDICAL ATTENTION — Go to the nearest Emergency Department or call 911/112.",
            "Do not drive yourself — ask someone to drive you or call an ambulance.",
            "While waiting for help: remain calm, stay seated, loosen restrictive clothing.",
            "Inform emergency responders of ALL your symptoms and any medications you take.",
            "Do not eat or drink anything until evaluated by a doctor.",
        ],
    }

    def __init__(self):
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0.3,
            openai_api_key=OPENAI_API_KEY,
        )
        logger.info("RecommendationAgent initialized")

    def generate_recommendations(
        self,
        symptoms: List[ExtractedSymptom],
        conditions: List[PossibleCondition],
        risk_level: RiskLevel,
        rag_context: str,
    ) -> List[str]:
        """
        Generate personalized health recommendations.

        Args:
            symptoms: Extracted symptoms
            conditions: Possible conditions
            risk_level: Assessed risk level
            rag_context: RAG medical context

        Returns:
            List of actionable recommendation strings
        """
        symptom_str = ", ".join(s.name for s in symptoms) or "reported symptoms"
        condition_str = ", ".join(c.name for c in conditions[:3]) or "unknown"

        prompt = f"""Patient Risk Level: {risk_level.value.upper()}
Symptoms: {symptom_str}
Possible Conditions: {condition_str}

Medical Context:
{rag_context[:600]}

Generate 4-6 specific, actionable recommendations appropriate for this patient.
Consider:
1. Immediate actions (based on risk level)
2. Self-care measures that may help
3. When to escalate care
4. Red flags to watch for
5. Lifestyle considerations

{"START with: '🚨 CALL EMERGENCY SERVICES / GO TO ER IMMEDIATELY'" if risk_level == RiskLevel.HIGH else ""}
{"START with: '⚠️ You should see a doctor within 24 hours'" if risk_level == RiskLevel.MODERATE else ""}

Numbered list, plain language, specific and actionable:"""

        try:
            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]

            response = self.llm.invoke(messages)
            content = response.content.strip()

            # Parse numbered list
            recommendations = []
            for line in content.split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith(("🚨", "⚠️", "-", "•"))):
                    cleaned = line.lstrip("0123456789.-•) ").strip()
                    if len(cleaned) > 10:  # Filter out very short lines
                        recommendations.append(cleaned)

            if recommendations:
                logger.info(f"Generated {len(recommendations)} recommendations")
                return recommendations[:6]
            else:
                return self.BASELINE_RECOMMENDATIONS.get(risk_level, [])

        except Exception as e:
            logger.error(f"RecommendationAgent error: {e}")
            return self.BASELINE_RECOMMENDATIONS.get(risk_level, [])

    def generate_explanation(
        self,
        symptoms: List[ExtractedSymptom],
        conditions: List[PossibleCondition],
        risk_level: RiskLevel,
        reasoning_summary: str,
    ) -> str:
        """
        Generate a plain-language explanation of the analysis.

        Args:
            symptoms: Extracted symptoms
            conditions: Possible conditions
            risk_level: Risk level
            reasoning_summary: Summary of reasoning chain

        Returns:
            Plain-language explanation string
        """
        symptom_str = ", ".join(s.name for s in symptoms) or "your symptoms"
        top_conditions = [c.name for c in conditions[:2]]
        conditions_str = " or ".join(top_conditions) if top_conditions else "require further evaluation"

        prompt = f"""Write a compassionate, plain-language explanation (3-5 sentences) for a patient about their health assessment.

Symptoms they reported: {symptom_str}
Most likely conditions: {conditions_str}  
Risk level: {risk_level.value.upper()}
Reasoning summary: {reasoning_summary[:300] if reasoning_summary else "Standard assessment"}

The explanation should:
- Use simple words (no medical jargon)
- Explain WHY these conditions were considered (symptom match)
- Be reassuring but honest about risk
- Not include specific medication advice
- End with a recommendation to see a doctor

Write the explanation:"""

        try:
            messages = [
                SystemMessage(content="You are a health educator explaining medical assessments to patients in plain, simple language."),
                HumanMessage(content=prompt),
            ]

            response = self.llm.invoke(messages)
            return response.content.strip()

        except Exception as e:
            logger.error(f"Explanation generation error: {e}")
            return (
                f"Based on your reported symptoms ({symptom_str}), our system identified "
                f"possible conditions including {conditions_str}. "
                f"The overall risk level has been assessed as {risk_level.value}. "
                "Please consult a healthcare professional for proper evaluation and diagnosis."
            )
