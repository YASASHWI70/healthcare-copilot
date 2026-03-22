"""
agents/symptom_extraction_agent.py - Extracts structured symptoms from free-text input.

Uses an LLM with a structured JSON output prompt to parse:
  - Symptom names
  - Severity (mild/moderate/severe)
  - Duration
  - Location (body part)
  - Modifiers (intermittent, worsening, etc.)
"""

# import json
# import re
# from typing import List, Optional

# from langchain_openai import ChatOpenAI
# from langchain_core.messages import SystemMessage, HumanMessage

# from backend.config import OPENAI_API_KEY, OPENAI_MODEL
# from backend.utils.models import ExtractedSymptom
# from backend.utils.logger import get_logger

logger = get_logger(__name__)


class SymptomExtractionAgent:
    """
    Extracts structured medical symptoms from unstructured patient text.

    Converts natural language descriptions like "I've had a bad headache
    and fever for two days" into structured ExtractedSymptom objects.
    """

    SYSTEM_PROMPT = """You are a medical NLP specialist. Your ONLY job is to extract symptoms 
from patient text and return structured JSON. 

RULES:
- Extract every symptom mentioned, even if vague
- Infer severity from descriptors (e.g., "terrible" = severe, "mild" = mild)
- Extract duration if mentioned
- Extract body location if mentioned
- List modifiers (e.g., "intermittent", "worsening", "with exertion", "at night")
- Return ONLY valid JSON — no preamble, no explanation, no markdown

Output format (strict JSON array):
[
  {
    "name": "symptom name",
    "severity": "mild | moderate | severe | null",
    "duration": "timeframe or null",
    "location": "body part or null",
    "modifiers": ["list", "of", "modifiers"]
  }
]"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0.0,  # Zero temp for consistent extraction
            openai_api_key=OPENAI_API_KEY,
        )
        logger.info("SymptomExtractionAgent initialized")

    def extract(self, patient_text: str, pdf_text: Optional[str] = None) -> List[ExtractedSymptom]:
        """
        Extract structured symptoms from patient input.

        Args:
            patient_text: Free-text symptom description from the user
            pdf_text: Optional extracted text from uploaded medical PDF

        Returns:
            List of ExtractedSymptom objects
        """
        if not patient_text.strip():
            return []

        # Combine inputs
        full_text = patient_text
        if pdf_text:
            full_text += f"\n\n[Medical Report Excerpt]:\n{pdf_text[:2000]}"  # Limit PDF text

        try:
            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=f"Extract symptoms from this patient text:\n\n{full_text}"),
            ]

            response = self.llm.invoke(messages)
            raw = response.content.strip()

            # Strip markdown code blocks if present
            raw = re.sub(r"```json|```", "", raw).strip()

            symptoms_data = json.loads(raw)

            symptoms = []
            for item in symptoms_data:
                symptom = ExtractedSymptom(
                    name=item.get("name", "").lower().strip(),
                    severity=item.get("severity"),
                    duration=item.get("duration"),
                    location=item.get("location"),
                    modifiers=item.get("modifiers", []),
                )
                if symptom.name:  # Skip empty names
                    symptoms.append(symptom)

            logger.info(f"Extracted {len(symptoms)} symptoms from patient text")
            return symptoms

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in symptom extraction: {e}")
            return self._fallback_extraction(patient_text)
        except Exception as e:
            logger.error(f"SymptomExtractionAgent error: {e}")
            return self._fallback_extraction(patient_text)

    def _fallback_extraction(self, text: str) -> List[ExtractedSymptom]:
        """
        Simple keyword-based fallback when LLM extraction fails.

        Args:
            text: Patient text

        Returns:
            Basic list of ExtractedSymptom objects
        """
        logger.warning("Using fallback keyword-based symptom extraction")

        COMMON_SYMPTOMS = [
            "fever", "cough", "headache", "fatigue", "nausea", "vomiting",
            "diarrhea", "chest pain", "shortness of breath", "sore throat",
            "runny nose", "body aches", "muscle pain", "joint pain",
            "dizziness", "rash", "sweating", "chills", "loss of appetite",
            "abdominal pain", "back pain", "swelling", "weakness",
        ]

        text_lower = text.lower()
        found = []
        for symptom in COMMON_SYMPTOMS:
            if symptom in text_lower:
                found.append(ExtractedSymptom(name=symptom))

        return found

    def get_symptom_names(self, symptoms: List[ExtractedSymptom]) -> List[str]:
        """Helper to extract just symptom names."""
        return [s.name for s in symptoms]
