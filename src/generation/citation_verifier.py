from __future__ import annotations

import re

from src.utils.logger import get_logger

_logger = get_logger(__name__)

_CITATION_RE = re.compile(r'\[Source:\s*([^\|]+?)\s*\|[^\]]+\]')


def verify_citations(
    response_text: str,
    included_citations: list[str],
    query_id: str = "",
) -> dict:
    """Verify that inline citations in *response_text* are grounded.

    Args:
        response_text:       The generated answer text.
        included_citations:  Citations returned by context_assembler (or ["visual"]).
        query_id:            Optional query identifier for warning logs.

    Returns:
        A dict with keys: path, verified, hallucinated (and note for visual path).
    """
    # Special case: visual path — no citation regex needed
    if included_citations == ["visual"]:
        return {
            "path": "visual",
            "verified": True,
            "hallucinated": [],
            "note": "visual path, source is rendered page",
        }

    # 1. Find all inline citation titles
    found_titles = [m.group(1).strip() for m in _CITATION_RE.finditer(response_text)]

    # 2. Classify each found title
    verified: list[str] = []
    hallucinated: list[str] = []

    included_lower = [c.lower() for c in included_citations]

    for title in found_titles:
        title_lower = title.lower()
        # Match if either direction is a substring — handles cases where the
        # model writes a longer title than the stored metadata (e.g. adds author)
        if any(title_lower in c or c in title_lower for c in included_lower):
            verified.append(title)
        else:
            hallucinated.append(title)

    # 4. Log
    _logger.info(
        "verify_citations",
        extra={
            "verified_count": len(verified),
            "hallucinated_count": len(hallucinated),
        },
    )

    if hallucinated:
        _logger.warning(
            "hallucinated_citations",
            extra={
                "query_id": query_id,
                "hallucinated": hallucinated,
            },
        )

    # 3. Return
    return {
        "path": "text",
        "verified": verified,
        "hallucinated": hallucinated,
    }
