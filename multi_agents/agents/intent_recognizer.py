"""Intent recognition helpers for query analysis and feedback expansion."""
from __future__ import annotations

from gpt_researcher.config.config import Config

from .utils.llms import call_model

_ALLOWED_QUERY_INTENTS = frozenset({
    "analytical",
    "descriptive",
    "comparative",
    "exploratory",
})

_REVIEW_SYSTEM_PROMPT = (
    "You are a research article review advisor. "
    "The user has read the reviewer's feedback on a research article draft and wants to add their own opinion. "
    "Your task is to expand the user's brief comment into a clear, actionable revision instruction.\n\n"
    "The expanded instruction must specify:\n"
    "1. Which section or aspect of the article to change\n"
    "2. What the specific change should be (add, remove, rewrite, restructure, etc.)\n"
    "3. Why this change improves the article\n\n"
    "Output only the expanded instruction, no preamble and no explanation. "
    "Reply in the same language as the user feedback."
)

_SYSTEM_PROMPT = (
    "You are a research outline revision advisor. "
    "The user has reviewed a research outline and provided brief feedback. "
    "Your task is to expand that feedback into a complete, actionable revision instruction.\n\n"
    "The expanded instruction must specify:\n"
    "1. Which sections to add, remove, or modify (by name)\n"
    "2. What sub-topics or angles each change should cover\n"
    "3. Suggested research focus for any new sections\n\n"
    "Output only the expanded instruction, no preamble and no explanation. "
    "Reply in the same language as the user feedback."
)

_QUERY_SYSTEM_PROMPT = (
    "You are a research query intent recognizer. "
    "Analyze the user's initial research query and return a compact JSON object.\n\n"
    "Classify the query intent as exactly one of: analytical, descriptive, comparative, exploratory.\n"
    "Also produce a normalized_query that keeps the original meaning but removes ambiguity and noise.\n"
    "Also produce a concise research_goal that states what the research should help the user understand.\n\n"
    "Return JSON only with this shape:\n"
    "{\n"
    '  "query_intent": "analytical|descriptive|comparative|exploratory",\n'
    '  "normalized_query": "string",\n'
    '  "research_goal": "string"\n'
    "}"
)


class IntentRecognizer:
    """Expand revision feedback and analyze the user's initial query."""

    def __init__(self, model: str | None = None) -> None:
        """Initialize the recognizer with a lightweight planning-capable model."""
        cfg = Config()
        self._model = model or getattr(cfg, "fast_llm_model", None) or cfg.smart_llm_model

    async def analyze_query(self, raw_query: str) -> dict[str, str]:
        """Classify one user query and return normalized query metadata."""
        cleaned_query = str(raw_query or "").strip()
        if not cleaned_query:
            return {
                "query_intent": "exploratory",
                "normalized_query": "",
                "research_goal": "",
            }

        prompt = [
            {"role": "system", "content": _QUERY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"User query: {cleaned_query}\n\n"
                    "Analyze this query and return the JSON object."
                ),
            },
        ]
        result = await call_model(
            prompt,
            model=self._model,
            response_format="json",
            route_context={
                "application_name": "auto_research_engine",
                "agent_role": "intent_recognizer",
                "stage_name": "query_intent_analysis",
                "task": cleaned_query,
            },
        ) or {}
        return self._normalize_query_analysis(result, cleaned_query)

    async def expand(self, raw_feedback: str, outline: str) -> str:
        """Return an LLM-expanded version of raw_feedback given outline context."""
        prompt = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Current outline:\n{outline}\n\n"
                    f"User feedback: {raw_feedback}\n\n"
                    "Expand this into a complete revision instruction."
                ),
            },
        ]
        result = await call_model(
            prompt,
            model=self._model,
            route_context={
                "application_name": "auto_research_engine",
                "agent_role": "intent_recognizer",
                "stage_name": "feedback_expansion",
            },
        )
        return str(result).strip()

    async def expand_review_feedback(
        self,
        raw_feedback: str,
        agent_review: str,
        draft_excerpt: str = "",
    ) -> str:
        """Expand one brief human review comment into an actionable instruction."""
        context = f"Reviewer's existing feedback:\n{agent_review}\n"
        if draft_excerpt:
            context += f"\nDraft excerpt (first 800 chars):\n{draft_excerpt[:800]}\n"
        prompt = [
            {"role": "system", "content": _REVIEW_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{context}\n"
                    f"User's additional comment: {raw_feedback}\n\n"
                    "Expand this into a complete revision instruction."
                ),
            },
        ]
        result = await call_model(
            prompt,
            model=self._model,
            route_context={
                "application_name": "auto_research_engine",
                "agent_role": "intent_recognizer",
                "stage_name": "review_feedback_expansion",
            },
        )
        return str(result).strip()

    @staticmethod
    def _normalize_query_analysis(result: dict[str, object], raw_query: str) -> dict[str, str]:
        """Normalize LLM query analysis output into a stable payload."""
        intent = str(result.get("query_intent") or "").strip().lower()
        if intent not in _ALLOWED_QUERY_INTENTS:
            intent = "exploratory"

        normalized_query = str(result.get("normalized_query") or "").strip() or raw_query
        research_goal = str(result.get("research_goal") or "").strip()
        return {
            "query_intent": intent,
            "normalized_query": normalized_query,
            "research_goal": research_goal,
        }
