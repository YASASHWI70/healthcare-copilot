"""
agents/conversation_agent.py - Manages user interaction and conversational context.

This agent is responsible for:
  - Maintaining a rolling conversation window
  - Generating natural-language follow-up questions
  - Formatting the final assistant message for the chat UI
"""

from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from backend.config import OPENAI_API_KEY, OPENAI_MODEL, MAX_CONVERSATION_HISTORY
from backend.utils.models import ChatMessage
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class ConversationAgent:
    """
    Manages conversational state and generates engaging, empathetic responses.

    Maintains a rolling window of conversation history to support
    multi-turn interactions and contextual follow-ups.
    """

    SYSTEM_PROMPT = """You are a compassionate, professional AI healthcare assistant called MediCopilot.
Your role is to help users understand their symptoms and guide them toward appropriate care.

IMPORTANT RULES:
1. You are NOT a doctor and cannot diagnose or prescribe
2. Always maintain a calm, reassuring, professional tone
3. Be empathetic — patients may be anxious
4. Use simple, clear language (no unnecessary medical jargon)
5. Always recommend professional medical consultation for anything beyond minor symptoms
6. For emergencies, ALWAYS tell users to call emergency services immediately
7. Keep responses focused and structured

You work as part of a multi-agent system. Your role is to:
- Greet and orient the user
- Ask clarifying questions when needed
- Synthesize agent outputs into a warm, human response
- Guide the conversation flow naturally"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0.3,
            openai_api_key=OPENAI_API_KEY,
        )
        logger.info("ConversationAgent initialized")

    def _build_messages(
        self,
        user_message: str,
        conversation_history: List[ChatMessage],
        context: str = "",
    ) -> List:
        """
        Build the message list for the LLM call.

        Args:
            user_message: Current user input
            conversation_history: Prior conversation turns
            context: Optional additional context to inject

        Returns:
            List of LangChain message objects
        """
        messages = [SystemMessage(content=self.SYSTEM_PROMPT)]

        # Add rolling history (limited window)
        recent_history = conversation_history[-MAX_CONVERSATION_HISTORY:]
        for msg in recent_history:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))

        # Build current user message with optional context
        current_content = user_message
        if context:
            current_content = f"{user_message}\n\n[Analysis Context: {context}]"

        messages.append(HumanMessage(content=current_content))
        return messages

    def generate_conversational_reply(
        self,
        user_message: str,
        conversation_history: List[ChatMessage],
        analysis_summary: str = "",
    ) -> str:
        """
        Generate a warm, conversational reply that wraps the clinical analysis.

        Args:
            user_message: Original user input
            conversation_history: Conversation history
            analysis_summary: Summary of findings from other agents

        Returns:
            Friendly conversational reply string
        """
        try:
            prompt = f"""The user said: "{user_message}"

Based on the medical analysis completed by our system, here is a summary of findings:
{analysis_summary}

Please write a warm, empathetic conversational response that:
1. Acknowledges the user's concern
2. Briefly summarizes the key findings in plain language
3. Reminds them this is not medical advice
4. Encourages them to seek professional care if needed
5. Is conversational, not clinical/robotic

Keep it concise (3-5 sentences max)."""

            messages = self._build_messages(prompt, conversation_history)
            response = self.llm.invoke(messages)
            return response.content

        except Exception as e:
            logger.error(f"ConversationAgent error: {e}")
            return (
                "I've analyzed your symptoms and prepared a detailed report below. "
                "Please review the findings and remember to consult a healthcare "
                "professional for proper evaluation."
            )

    def generate_follow_up_questions(
        self,
        extracted_symptoms: List[str],
        possible_conditions: List[str],
    ) -> List[str]:
        """
        Generate 2-3 clinically relevant follow-up questions.

        Args:
            extracted_symptoms: List of symptom names
            possible_conditions: List of possible condition names

        Returns:
            List of follow-up question strings
        """
        try:
            symptoms_str = ", ".join(extracted_symptoms) if extracted_symptoms else "the reported symptoms"
            conditions_str = ", ".join(possible_conditions) if possible_conditions else "the possible conditions"

            prompt = f"""A patient reported: {symptoms_str}
Possible conditions under consideration: {conditions_str}

Generate exactly 3 short, specific follow-up questions that would help narrow down the diagnosis.
Questions should be:
- Clinically relevant (would change differential)
- Easy for a non-medical person to understand
- Not yes/no only questions (encourage descriptive answers)
- Each on a new line, numbered 1-3
- No preamble, just the questions"""

            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]
            response = self.llm.invoke(messages)

            # Parse numbered questions
            lines = response.content.strip().split("\n")
            questions = []
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    # Remove leading number/dash
                    cleaned = line.lstrip("0123456789.-) ").strip()
                    if cleaned:
                        questions.append(cleaned)

            return questions[:3]  # Ensure max 3

        except Exception as e:
            logger.error(f"Follow-up question generation error: {e}")
            return [
                "How long have you been experiencing these symptoms?",
                "Have you taken any medications for these symptoms?",
                "Do you have any pre-existing medical conditions?",
            ]
