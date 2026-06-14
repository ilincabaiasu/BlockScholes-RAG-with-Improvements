from __future__ import annotations

import json
import re

from src.config.gemini_client import generate_text
from src.utils.logger import get_logger

_logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a query rewriter for a Block Scholes crypto research RAG.\n"
    "Rewrite the user query to improve retrieval from research documents.\n"
    "Rules:\n"
    "  - Expand acronyms (BTC→Bitcoin, ETH→Ethereum, IV→implied volatility)\n"
    "  - Fully qualify asset names and timeframes\n"
    "  - If the query has multiple questions, split into sub_queries\n"
    'Return ONLY valid JSON in this exact format:\n'
    '{"rewritten_query": "...", "sub_queries": ["...", "..."]}\n'
    "No markdown, no explanation."
)

# Strip markdown code fences if the model wraps its output
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _fallback(query: str) -> dict:
    return {"rewritten_query": query, "sub_queries": [query]}


async def rewrite_query(query: str) -> dict:
    """Rewrite *query* for improved retrieval and split into sub-queries.

    Returns:
        {"rewritten_query": str, "sub_queries": list[str]}
    Falls back to the original query on any parse or validation failure.
    """
    try:
        raw = await generate_text(_SYSTEM_PROMPT, query, temperature=0.0)

        # Strip markdown code fences if present
        fence_match = _FENCE_RE.search(raw)
        text = fence_match.group(1) if fence_match else raw.strip()

        parsed = json.loads(text)

        rewritten = parsed.get("rewritten_query", "")
        sub_queries = parsed.get("sub_queries", [])

        if (
            not isinstance(rewritten, str)
            or not rewritten.strip()
            or not isinstance(sub_queries, list)
            or not sub_queries
            or not all(isinstance(s, str) for s in sub_queries)
        ):
            raise ValueError(f"validation failed: {parsed}")

        result = {"rewritten_query": rewritten, "sub_queries": sub_queries}

        _logger.info(
            "rewrite_query",
            extra={
                "original_query": query[:80],
                "rewritten_query": rewritten[:80],
                "sub_query_count": len(sub_queries),
            },
        )

        return result

    except Exception as exc:
        _logger.warning(
            "rewriter_fallback",
            extra={"error": str(exc), "query": query[:80]},
        )
        return _fallback(query)
