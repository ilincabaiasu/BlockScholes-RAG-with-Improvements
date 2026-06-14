from __future__ import annotations

from src.retrieval.dense_retriever import ScoredChunk
from src.utils.logger import get_logger

_logger = get_logger(__name__)


def route(
    chunks: list[ScoredChunk],
) -> tuple[list[ScoredChunk], list[ScoredChunk]]:
    """Split chunks into text and visual paths.

    A chunk is visual if its payload has has_visual_content=True
    AND a non-empty pdf_path (required for page rendering).

    Returns:
        (text_chunks, visual_chunks)
    """
    text_chunks: list[ScoredChunk] = []
    visual_chunks: list[ScoredChunk] = []

    for chunk in chunks:
        if (
            chunk.metadata.get("has_visual_content", False)
            and chunk.metadata.get("pdf_path", "")
        ):
            visual_chunks.append(chunk)
        else:
            text_chunks.append(chunk)

    _logger.info(
        "visual_route",
        extra={
            "total": len(chunks),
            "text_count": len(text_chunks),
            "visual_count": len(visual_chunks),
        },
    )

    return text_chunks, visual_chunks
