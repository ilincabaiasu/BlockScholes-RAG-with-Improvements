from __future__ import annotations

from src.config.gemini_client import generate_text
from src.utils.logger import get_logger

_logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a query classifier for a financial research RAG system.\n"
    "Classify the user query into exactly one category.\n"
    "Reply with ONLY the category name, nothing else.\n"
    "Categories:\n"
    "  factual_lookup  – asks for a specific fact or number\n"
    "  analytical      – asks for explanation, interpretation, or insight\n"
    "  comparative     – asks to compare two things or time periods\n"
    "  definitional    – asks what something means or how it works"
)

_VALID_CATEGORIES = frozenset(
    {"factual_lookup", "analytical", "comparative", "definitional"}
)


async def classify_query(query: str) -> str:
    """Classify *query* into one of four categories using Gemini.

    Returns one of: "factual_lookup", "analytical", "comparative", "definitional".
    Falls back to "analytical" on any error or unexpected response.
    """
    try:
        response = await generate_text(_SYSTEM_PROMPT, query, temperature=0.0)
        category = response.strip()
        if category in _VALID_CATEGORIES:
            return category
        _logger.warning(
            "classifier_unexpected_response",
            extra={"response": category, "query": query[:80]},
        )
    except Exception as exc:
        _logger.warning(
            "classifier_error",
            extra={"error": str(exc), "query": query[:80]},
        )

    return "analytical"
