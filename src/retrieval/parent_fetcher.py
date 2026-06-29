from __future__ import annotations

from src.config.settings import settings
from src.ingestion.models import ContextChunk
from src.retrieval.dense_retriever import ScoredChunk
from src.utils.logger import get_logger
from src.utils.token_counter import count_tokens

_logger = get_logger(__name__)


async def fetch_parents(
    chunks: list[ScoredChunk],
    max_context_tokens: int | None = None,
    use_parent: bool = True,
) -> list[ContextChunk]:
    """Build ContextChunk objects from reranked ScoredChunks.

    For each child chunk, uses the parent_text stored in its Qdrant payload
    (populated at upsert time). Parent chunks are not stored as separate
    Qdrant points — parent_text lives in chunk.metadata["parent_text"].

    Respects a token budget: if adding the parent text would exceed
    max_context_tokens, falls back to the child text only.

    Args:
        chunks:             ScoredChunks sorted by reranker_score descending.
        max_context_tokens: Token budget; defaults to settings.MAX_CONTEXT_TOKENS.
        use_parent:         When False (ablation), the hierarchical expansion is
                            skipped and only the child-chunk text is used.

    Returns:
        List of ContextChunk objects in the same order.
    """
    if max_context_tokens is None:
        max_context_tokens = settings.MAX_CONTEXT_TOKENS

    total_tokens = 0
    context_chunks: list[ContextChunk] = []

    for chunk in chunks:
        # parent_text is pre-stored in the child payload at index time. With
        # parent fetching ablated, use the child text directly.
        if use_parent:
            parent_text = chunk.metadata.get("parent_text") or chunk.text
        else:
            parent_text = chunk.text
        parent_token_count = count_tokens(parent_text)

        if total_tokens + parent_token_count > max_context_tokens:
            # Fall back to child text to stay within budget
            child_token_count = count_tokens(chunk.text)
            used_text = chunk.text
            _logger.info(
                "truncated_to_child",
                extra={
                    "chunk_id": chunk.chunk_id,
                    "parent_tokens": parent_token_count,
                    "budget_remaining": max(0, max_context_tokens - total_tokens - child_token_count),
                },
            )
            total_tokens += child_token_count
        else:
            used_text = parent_text
            total_tokens += parent_token_count

        context_chunks.append(
            ContextChunk(
                parent_text=used_text,
                child_text=chunk.text,
                metadata=chunk.metadata,
                reranker_score=chunk.reranker_score,
                visual_path_used=False,
                source_title=chunk.metadata.get("title", ""),
                source_date=chunk.metadata.get("published_date", ""),
            )
        )

    _logger.info(
        "fetch_parents",
        extra={
            "input_chunks": len(chunks),
            "output_chunks": len(context_chunks),
            "total_tokens": total_tokens,
        },
    )

    return context_chunks
