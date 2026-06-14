from __future__ import annotations

import time
from dataclasses import replace

from src.config.cohere_client import rerank_with_retry
from src.config.settings import settings
from src.retrieval.dense_retriever import ScoredChunk
from src.utils.logger import get_logger

_logger = get_logger(__name__)


async def rerank(
    query: str,
    candidates: list[ScoredChunk],
    top_k: int | None = None,
) -> list[ScoredChunk]:
    """Rerank *candidates* using Cohere cross-encoder.

    Args:
        query:      The original query string.
        candidates: Chunks to rerank (typically from hybrid merge).
        top_k:      Max results to return; defaults to settings.RERANKER_TOP_K.

    Returns:
        List of ScoredChunk with reranker_score set, sorted descending,
        filtered to scores >= settings.RERANKER_MIN_SCORE.
    """
    if top_k is None:
        top_k = settings.RERANKER_TOP_K

    if not candidates:
        return []

    t0 = time.perf_counter()

    documents = [c.text for c in candidates]

    response = await rerank_with_retry(
        query=query,
        documents=documents,
        top_n=len(candidates),
        model=settings.RERANKER_MODEL,
    )

    # Map reranker results back to ScoredChunk objects by index
    scored: list[ScoredChunk] = []
    for result in response.results:
        chunk = candidates[result.index]
        scored.append(replace(chunk, reranker_score=result.relevance_score))

    # Filter by minimum score
    before_filter = len(scored)
    scored = [c for c in scored if c.reranker_score >= settings.RERANKER_MIN_SCORE]
    dropped = before_filter - len(scored)
    if dropped:
        _logger.info(
            "reranker_filtered",
            extra={"dropped": dropped, "min_score": settings.RERANKER_MIN_SCORE},
        )

    scored.sort(key=lambda c: c.reranker_score, reverse=True)
    output = scored[:top_k]

    elapsed_ms = (time.perf_counter() - t0) * 1_000
    _logger.info(
        "rerank",
        extra={
            "input_count": len(candidates),
            "output_count": len(output),
            "top_score": round(output[0].reranker_score, 4) if output else 0.0,
            "elapsed_ms": round(elapsed_ms, 3),
        },
    )

    return output
