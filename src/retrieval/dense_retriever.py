from __future__ import annotations

import time
from dataclasses import dataclass

from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.config.qdrant_client import qdrant
from src.config.settings import settings
from src.embedding.embedder import embed_query
from src.utils.logger import get_logger

_logger = get_logger(__name__)


@dataclass
class ScoredChunk:
    chunk_id: str
    text: str
    parent_id: str
    metadata: dict
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    reranker_score: float = 0.0


def _build_filter(filters: dict) -> Filter | None:
    """Build a Qdrant Filter from a dict of supported keys."""
    if not filters:
        return None
    conditions = [
        FieldCondition(key=k, match=MatchValue(value=v))
        for k, v in filters.items()
        if k in ("doc_type", "asset_class")
    ]
    return Filter(must=conditions) if conditions else None


async def dense_search(
    query_text: str,
    top_k: int | None = None,
    filters: dict | None = None,
) -> list[ScoredChunk]:
    """Search Qdrant by dense vector similarity.

    Args:
        query_text: The query string to embed and search with.
        top_k:      Number of results to return; defaults to settings.DENSE_TOP_K.
        filters:    Optional dict with keys "doc_type" and/or "asset_class".

    Returns:
        List of ScoredChunk objects sorted by dense_score descending.
    """
    if top_k is None:
        top_k = settings.DENSE_TOP_K

    t0 = time.perf_counter()

    query_vector = await embed_query(query_text)
    query_filter = _build_filter(filters or {})

    try:
        response = await qdrant.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
    except Exception as exc:
        _logger.error("dense_search_failed", extra={"error": str(exc)})
        raise

    chunks: list[ScoredChunk] = []
    for r in response.points:
        p = r.payload or {}
        chunk_id = p.get("chunk_id", "")
        if not chunk_id:
            _logger.warning("dense_result_missing_chunk_id", extra={"point_id": r.id})
            continue
        chunks.append(
            ScoredChunk(
                chunk_id=chunk_id,
                text=p.get("text", ""),
                parent_id=p.get("parent_id", ""),
                metadata=p,
                dense_score=r.score,
            )
        )

    elapsed_ms = (time.perf_counter() - t0) * 1_000
    _logger.info(
        "dense_search",
        extra={
            "query_len": len(query_text),
            "results": len(chunks),
            "elapsed_ms": round(elapsed_ms, 3),
        },
    )

    return chunks
