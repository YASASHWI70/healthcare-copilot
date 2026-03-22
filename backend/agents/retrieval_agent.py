"""
agents/retrieval_agent.py - RAG agent that fetches relevant medical knowledge.

Queries the FAISS vector store with symptom-based queries and
returns grounding context for other agents to use.
"""

# from typing import List
# from backend.rag.vector_store import retrieve_relevant_context
# from backend.utils.models import ExtractedSymptom
# from backend.utils.logger import get_logger

# logger = get_logger(__name__)


class RetrievalAgent:
    """
    Retrieval-Augmented Generation (RAG) agent.

    Converts extracted symptoms into semantic queries and retrieves
    relevant medical knowledge from the FAISS vector database.
    This grounding context ensures LLM responses are anchored
    in real medical knowledge rather than pure hallucination.
    """

    def __init__(self):
        logger.info("RetrievalAgent initialized")

    def retrieve(
        self,
        symptoms: List[ExtractedSymptom],
        user_query: str,
        top_k: int = 5,
    ) -> List[str]:
        """
        Retrieve relevant medical context for given symptoms.

        Strategy:
          1. Build a symptom-focused query
          2. Also query with the original user text for semantic coverage
          3. Deduplicate results
          4. Return top snippets

        Args:
            symptoms: Extracted symptom objects
            user_query: Original user input for fallback query
            top_k: Maximum number of snippets to return

        Returns:
            List of relevant medical knowledge snippets
        """
        if not symptoms and not user_query:
            return []

        all_snippets: List[str] = []
        seen: set = set()

        # Query 1: Symptom-focused query
        if symptoms:
            symptom_names = [s.name for s in symptoms]
            symptom_query = f"symptoms: {', '.join(symptom_names)}"

            # Add severity context if available
            severe_symptoms = [s.name for s in symptoms if s.severity == "severe"]
            if severe_symptoms:
                symptom_query += f" | severe: {', '.join(severe_symptoms)}"

            snippets = retrieve_relevant_context(symptom_query, top_k=top_k)
            for snippet in snippets:
                if snippet not in seen:
                    seen.add(snippet)
                    all_snippets.append(snippet)

        # Query 2: Original user text (semantic)
        if user_query:
            snippets = retrieve_relevant_context(user_query, top_k=3)
            for snippet in snippets:
                if snippet not in seen:
                    seen.add(snippet)
                    all_snippets.append(snippet)

        # Query 3: Risk-focused query for comprehensive coverage
        risk_query = "red flags emergency high risk symptoms when to seek care"
        snippets = retrieve_relevant_context(risk_query, top_k=2)
        for snippet in snippets:
            if snippet not in seen:
                seen.add(snippet)
                all_snippets.append(snippet)

        # Limit total context
        result = all_snippets[:top_k]
        logger.info(f"RetrievalAgent retrieved {len(result)} context snippets")
        return result

    def format_context_for_prompt(self, snippets: List[str]) -> str:
        """
        Format retrieved snippets into a prompt-friendly string.

        Args:
            snippets: List of retrieved text snippets

        Returns:
            Formatted context block
        """
        if not snippets:
            return "No specific medical knowledge retrieved."

        context_parts = []
        for i, snippet in enumerate(snippets, 1):
            context_parts.append(f"[Context {i}]:\n{snippet}")

        return "\n\n".join(context_parts)
