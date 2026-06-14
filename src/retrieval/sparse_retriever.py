from __future__ import annotations

import time

from src.config.qdrant_client import qdrant
from src.config.settings import settings
from src.indexing.bm25_index import search as bm25_search
from src.indexing.upserter import chunk_uuid
from src.retrieval.dense_retriever import ScoredChunk
from src.utils.logger import get_logger

_logger = get_logger(__name__)


async def sparse_search(
    query_text: str,
    top_k: int | None = None,
) -> list[ScoredChunk]:
    """Search using BM25 and hydrate results with Qdrant payloads.

    Args:
        query_text: The query string.
        top_k:      Number of results; defaults to settings.SPARSE_TOP_K.

    Returns:
        List of ScoredChunk objects with sparse_score set.
    """
    if top_k is None:
        top_k = settings.SPARSE_TOP_K

    t0 = time.perf_counter()

    # 1. BM25 retrieval — returns (chunk_id, score) tuples
    bm25_results: list[tuple[str, float]] = bm25_search(query_text, top_k)

    if not bm25_results:
        return []

    # 2. Batch-retrieve payloads from Qdrant by UUID point IDs
    chunk_ids = [cid for cid, _ in bm25_results]
    score_by_id = {cid: score for cid, score in bm25_results}
    point_ids = [chunk_uuid(cid) for cid in chunk_ids]

    try:
        records = await qdrant.retrieve(
            collection_name=settings.QDRANT_COLLECTION,
            ids=point_ids,
            with_payload=True,
        )
    except Exception as exc:
        _logger.error("sparse_search_failed", extra={"error": str(exc)})
        raise

    # Build lookup: uuid → payload
    payload_by_uuid = {str(r.id): r.payload for r in records if r.payload}

    # 3. Build ScoredChunk list, preserving BM25 rank order
    chunks: list[ScoredChunk] = []
    for chunk_id in chunk_ids:
        point_uuid = chunk_uuid(chunk_id)
        payload = payload_by_uuid.get(point_uuid)
        if payload is None:
            _logger.warning(
                "sparse_chunk_not_found",
                extra={"chunk_id": chunk_id},
            )
            continue
        chunks.append(
            ScoredChunk(
                chunk_id=payload.get("chunk_id", chunk_id),
                text=payload.get("text", ""),
                parent_id=payload.get("parent_id", ""),
                metadata=payload,
                sparse_score=score_by_id[chunk_id],
            )
        )

    elapsed_ms = (time.perf_counter() - t0) * 1_000
    _logger.info(
        "sparse_search",
        extra={
            "query_len": len(query_text),
            "bm25_hits": len(bm25_results),
            "results": len(chunks),
            "elapsed_ms": round(elapsed_ms, 3),
        },
    )

    return chunks
