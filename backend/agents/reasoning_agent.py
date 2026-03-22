"""
agents/reasoning_agent.py - Performs step-by-step clinical reasoning using LLMs.

This is the core diagnostic reasoning engine that:
  1. Takes symptoms + RAG context as input
  2. Applies chain-of-thought reasoning
  3. Returns possible conditions with confidence levels
  4. Generates reasoning steps for explainability
"""

# import json
# import re
# from typing import List, Tuple

# from langchain_openai import ChatOpenAI
# from langchain_core.messages import SystemMessage, HumanMessage

# from backend.config import OPENAI_API_KEY, OPENAI_MODEL
# from backend.utils.models import ExtractedSymptom, PossibleCondition, ReasoningStep
# from backend.utils.logger import get_logger

logger = get_logger(__name__)


class ReasoningAgent:
    """
    Medical reasoning agent using chain-of-thought LLM prompting.

    Implements a structured reasoning approach:
    Symptoms → Pattern Recognition → Differential Diagnosis → Ranked Conditions
    """

    SYSTEM_PROMPT = """You are a clinical reasoning AI assistant. Your role is to perform 
structured medical reasoning to identify possible conditions from symptoms.

IMPORTANT GUIDELINES:
- Use evidence-based differential diagnosis approach
- Consider common conditions first (horses before zebras)
- Always flag if any RED FLAG symptoms are present
- Be conservative — err on the side of caution
- You are NOT making a diagnosis; you are generating a differential for educational purposes
- Consider the provided medical knowledge context carefully

You MUST return valid JSON only — no preamble, no explanation outside the JSON."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0.1,
            openai_api_key=OPENAI_API_KEY,
        )
        logger.info("ReasoningAgent initialized")

    def reason(
        self,
        symptoms: List[ExtractedSymptom],
        rag_context: str,
        user_query: str,
    ) -> Tuple[List[PossibleCondition], List[ReasoningStep]]:
        """
        Perform chain-of-thought medical reasoning.

        Args:
            symptoms: Structured extracted symptoms
            rag_context: Relevant medical knowledge from RAG
            user_query: Original patient description

        Returns:
            Tuple of (possible_conditions, reasoning_steps)
        """
        if not symptoms:
            return [], []

        # Build structured symptom description
        symptom_descriptions = []
        for s in symptoms:
            desc = s.name
            if s.severity:
                desc += f" ({s.severity})"
            if s.duration:
                desc += f" for {s.duration}"
            if s.location:
                desc += f" in {s.location}"
            if s.modifiers:
                desc += f" [{', '.join(s.modifiers)}]"
            symptom_descriptions.append(desc)

        symptoms_str = "\n".join(f"- {d}" for d in symptom_descriptions)

        prompt = f"""PATIENT SYMPTOMS:
{symptoms_str}

ORIGINAL PATIENT DESCRIPTION:
"{user_query}"

MEDICAL KNOWLEDGE CONTEXT:
{rag_context}

---

Perform clinical reasoning and return a JSON object with this EXACT structure:
{{
  "reasoning_steps": [
    {{
      "step_number": 1,
      "title": "Symptom Pattern Analysis",
      "content": "Detailed analysis of the symptom pattern..."
    }},
    {{
      "step_number": 2,
      "title": "Differential Diagnosis Generation",
      "content": "Listing potential conditions that match this symptom cluster..."
    }},
    {{
      "step_number": 3,
      "title": "Red Flag Assessment",
      "content": "Identifying any concerning or emergency symptoms..."
    }},
    {{
      "step_number": 4,
      "title": "Probability Ranking",
      "content": "Ranking conditions by likelihood based on symptom fit..."
    }}
  ],
  "possible_conditions": [
    {{
      "name": "Condition Name",
      "confidence": "high | moderate | low",
      "matching_symptoms": ["symptom1", "symptom2"],
      "key_differentiators": "What makes this condition more or less likely"
    }}
  ]
}}

Rules:
- Include 2-5 possible conditions
- Order by confidence (highest first)
- Reasoning steps should be thorough (2-4 sentences each)
- If red flags present (chest pain, stroke symptoms, severe breathing difficulty), note prominently in Step 3
- Return ONLY the JSON object"""

        try:
            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]

            response = self.llm.invoke(messages)
            raw = response.content.strip()

            # Strip markdown code blocks
            raw = re.sub(r"```json|```", "", raw).strip()

            data = json.loads(raw)

            # Parse conditions
            conditions = []
            for c in data.get("possible_conditions", []):
                conditions.append(PossibleCondition(
                    name=c.get("name", "Unknown"),
                    confidence=c.get("confidence", "low"),
                    matching_symptoms=c.get("matching_symptoms", []),
                    key_differentiators=c.get("key_differentiators"),
                ))

            # Parse reasoning steps
            steps = []
            for s in data.get("reasoning_steps", []):
                steps.append(ReasoningStep(
                    step_number=s.get("step_number", 0),
                    title=s.get("title", ""),
                    content=s.get("content", ""),
                ))

            logger.info(f"ReasoningAgent identified {len(conditions)} possible conditions")
            return conditions, steps

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in reasoning: {e}")
            return self._fallback_reasoning(symptoms), []
        except Exception as e:
            logger.error(f"ReasoningAgent error: {e}")
            return [], []

    def _fallback_reasoning(
        self, symptoms: List[ExtractedSymptom]
    ) -> List[PossibleCondition]:
        """Fallback when LLM reasoning fails."""
        logger.warning("Using fallback condition mapping")

        SYMPTOM_CONDITION_MAP = {
            "fever": ("Viral Infection / Influenza", "moderate"),
            "cough": ("Upper Respiratory Infection", "moderate"),
            "headache": ("Tension Headache", "moderate"),
            "chest pain": ("Musculoskeletal Chest Pain (consider cardiac)", "high"),
            "shortness of breath": ("Respiratory Condition (urgent evaluation)", "high"),
            "nausea": ("Gastroenteritis", "low"),
            "diarrhea": ("Gastroenteritis", "low"),
            "sore throat": ("Pharyngitis / Tonsillitis", "moderate"),
            "fatigue": ("Viral Illness / Anemia", "low"),
        }

        conditions = []
        for symptom in symptoms:
            if symptom.name in SYMPTOM_CONDITION_MAP:
                name, confidence = SYMPTOM_CONDITION_MAP[symptom.name]
                if not any(c.name == name for c in conditions):
                    conditions.append(PossibleCondition(
                        name=name,
                        confidence=confidence,
                        matching_symptoms=[symptom.name],
                    ))

        return conditions[:3]
