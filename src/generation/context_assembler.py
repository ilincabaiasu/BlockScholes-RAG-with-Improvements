from __future__ import annotations

from src.config.settings import settings
from src.ingestion.models import ContextChunk
from src.utils.logger import get_logger
from src.utils.token_counter import count_tokens

_logger = get_logger(__name__)


def assemble_context(
    chunks: list[ContextChunk],
    max_context_tokens: int | None = None,
) -> tuple[str, list[str]]:
    """Assemble a token-budgeted context string from reranked chunks.

    Args:
        chunks: Reranked ContextChunk objects.

    Returns:
        A tuple of:
          - formatted context string (blocks joined by double newline)
          - deduplicated citation list in inclusion order
    """
    # 1. Sort: reranker_score desc, then published_date desc for ties
    sorted_chunks = sorted(
        chunks,
        key=lambda c: (c.reranker_score, c.source_date),
        reverse=True,
    )

    # 2. Running state
    running_tokens = 0
    included: list[str] = []
    citation_set: set[str] = set()
    citation_list: list[str] = []
    seen_parent_texts: set[int] = set()  # hash of parent_text to avoid duplicates

    budget = max_context_tokens if max_context_tokens is not None else settings.MAX_CONTEXT_TOKENS

    # 3. Fill budget
    for i, chunk in enumerate(sorted_chunks):
        # Skip if this exact parent text was already included
        parent_hash = hash(chunk.parent_text)
        if parent_hash in seen_parent_texts:
            continue
        seen_parent_texts.add(parent_hash)

        block = f"[Source: {chunk.source_title} | {chunk.source_date}]\n{chunk.parent_text}\n"
        block_tokens = count_tokens(block)

        if running_tokens + block_tokens > budget:
            dropped = len(sorted_chunks) - i
            _logger.info(
                "context_budget_reached",
                extra={"chunks_dropped": dropped},
            )
            break

        included.append(block)
        running_tokens += block_tokens

        citation = f"{chunk.source_title} ({chunk.source_date})"
        if citation not in citation_set:
            citation_set.add(citation)
            citation_list.append(citation)

    # 4. Return
    return "\n\n".join(included), citation_list
