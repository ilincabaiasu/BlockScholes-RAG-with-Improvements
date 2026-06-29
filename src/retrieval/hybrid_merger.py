from __future__ import annotations

from dataclasses import replace

from src.config.settings import settings
from src.retrieval.dense_retriever import ScoredChunk
from src.utils.logger import get_logger

_logger = get_logger(__name__)


def rrf_merge(
    dense_results: list[ScoredChunk],
    sparse_results: list[ScoredChunk],
) -> list[ScoredChunk]:
    """Merge dense and sparse results using Reciprocal Rank Fusion.

    Returns a single list of ScoredChunk sorted by rrf_score descending.
    """
    k = settings.RRF_K

    # Drop any results with empty chunk_id — they would collide in rank dicts
    # and corrupt fusion scores.
    dense_results = [c for c in dense_results if c.chunk_id]
    sparse_results = [c for c in sparse_results if c.chunk_id]

    # 1-indexed rank dicts
    dense_ranks = {c.chunk_id: i + 1 for i, c in enumerate(dense_results)}
    sparse_ranks = {c.chunk_id: i + 1 for i, c in enumerate(sparse_results)}

    # Lookup dicts for payload reuse
    dense_by_id = {c.chunk_id: c for c in dense_results}
    sparse_by_id = {c.chunk_id: c for c in sparse_results}

    all_ids = set(dense_ranks) | set(sparse_ranks)
    overlap_count = len(set(dense_ranks) & set(sparse_ranks))

    merged: list[ScoredChunk] = []
    for chunk_id in all_ids:
        rrf_score = 0.0
        if chunk_id in dense_ranks:
            rrf_score += 1.0 / (k + dense_ranks[chunk_id])
        if chunk_id in sparse_ranks:
            rrf_score += 1.0 / (k + sparse_ranks[chunk_id])

        # Prefer dense chunk (richer payload); fall back to sparse
        chunk = dense_by_id.get(chunk_id) or sparse_by_id[chunk_id]

        merged.append(replace(chunk, rrf_score=rrf_score))

    # Secondary sort on chunk_id makes tie-breaking deterministic across runs.
    merged.sort(key=lambda c: (c.rrf_score, c.chunk_id), reverse=True)

    _logger.info(
        "rrf_merge",
        extra={
            "dense_count": len(dense_results),
            "sparse_count": len(sparse_results),
            "merged_count": len(merged),
            "overlap_count": overlap_count,
        },
    )

    return merged
