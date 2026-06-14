from __future__ import annotations

from src.retrieval.dense_retriever import ScoredChunk
from src.retrieval.hybrid_merger import rrf_merge


def _make_chunk(chunk_id: str, dense_score: float = 0.0, sparse_score: float = 0.0) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        text=f"text for {chunk_id}",
        parent_id="",
        metadata={},
        dense_score=dense_score,
        sparse_score=sparse_score,
    )


def test_merge_count_with_one_overlap():
    # Dense: A(rank1), B(rank2), C(rank3)
    # Sparse: C(rank1), D(rank2), E(rank3)  — C is the overlap
    dense = [_make_chunk("A", dense_score=0.9),
             _make_chunk("B", dense_score=0.8),
             _make_chunk("C", dense_score=0.7)]
    sparse = [_make_chunk("C", sparse_score=0.9),
              _make_chunk("D", sparse_score=0.8),
              _make_chunk("E", sparse_score=0.7)]

    merged = rrf_merge(dense, sparse)
    assert len(merged) == 5


def test_overlapping_chunk_has_highest_rrf_score():
    dense = [_make_chunk("A", dense_score=0.9),
             _make_chunk("B", dense_score=0.8),
             _make_chunk("C", dense_score=0.7)]
    sparse = [_make_chunk("C", sparse_score=0.9),
              _make_chunk("D", sparse_score=0.8),
              _make_chunk("E", sparse_score=0.7)]

    merged = rrf_merge(dense, sparse)
    chunk_c = next(c for c in merged if c.chunk_id == "C")
    non_overlap_scores = [c.rrf_score for c in merged if c.chunk_id != "C"]
    assert all(chunk_c.rrf_score > s for s in non_overlap_scores)


def test_merge_sorted_descending():
    dense = [_make_chunk("A", dense_score=0.9),
             _make_chunk("B", dense_score=0.8),
             _make_chunk("C", dense_score=0.7)]
    sparse = [_make_chunk("C", sparse_score=0.9),
              _make_chunk("D", sparse_score=0.8),
              _make_chunk("E", sparse_score=0.7)]

    merged = rrf_merge(dense, sparse)
    scores = [c.rrf_score for c in merged]
    assert scores == sorted(scores, reverse=True)
