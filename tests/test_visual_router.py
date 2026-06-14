from __future__ import annotations

from src.retrieval.dense_retriever import ScoredChunk
from src.retrieval.visual_router import route


def _text_chunk(chunk_id: str) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        text="text",
        parent_id="",
        metadata={"has_visual_content": False, "pdf_path": ""},
    )


def _visual_chunk(chunk_id: str) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        text="text",
        parent_id="",
        metadata={"has_visual_content": True, "pdf_path": "/some/file.pdf"},
    )


def test_mixed_chunks_split_correctly():
    chunks = [_text_chunk("t1"), _text_chunk("t2"), _visual_chunk("v1")]
    text_chunks, visual_chunks = route(chunks)
    assert len(text_chunks) == 2
    assert len(visual_chunks) == 1
    assert visual_chunks[0].chunk_id == "v1"


def test_all_text_chunks():
    chunks = [_text_chunk("t1"), _text_chunk("t2"), _text_chunk("t3")]
    text_chunks, visual_chunks = route(chunks)
    assert text_chunks == chunks
    assert visual_chunks == []


def test_all_visual_chunks():
    chunks = [_visual_chunk("v1"), _visual_chunk("v2"), _visual_chunk("v3")]
    text_chunks, visual_chunks = route(chunks)
    assert text_chunks == []
    assert visual_chunks == chunks
