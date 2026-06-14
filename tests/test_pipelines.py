from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.pipelines.models import GenerationResult
from src.retrieval.dense_retriever import ScoredChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gen_result(**kwargs) -> GenerationResult:
    defaults = dict(response_text="mocked response", provider="gemini", model_name="gemini-2.5-flash")
    return GenerationResult(**{**defaults, **kwargs})


def _text_chunk(chunk_id: str = "chunk-1") -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        text="chunk text",
        parent_id="",
        metadata={
            "title": "Test Doc",
            "published_date": "2024-11-04",
            "parent_text": "parent text",
            "has_visual_content": False,
            "pdf_path": "",
        },
        dense_score=0.9,
        reranker_score=0.9,
    )


def _visual_chunk(chunk_id: str = "chunk-v1") -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        text="visual chunk text",
        parent_id="",
        metadata={
            "title": "Visual Doc",
            "published_date": "2024-11-04",
            "parent_text": "parent text",
            "has_visual_content": True,
            "pdf_path": "/data/raw/doc.pdf",
        },
        dense_score=0.8,
        reranker_score=0.8,
    )


# ---------------------------------------------------------------------------
# Gemini vanilla pipeline
# ---------------------------------------------------------------------------

async def test_gemini_pipeline_returns_valid_result():
    import src.pipelines.gemini_pipeline as gp

    with patch.object(gp, "generate_text", new=AsyncMock(return_value="answer")), \
         patch.object(gp, "log_result", new=MagicMock()):
        result = await gp.run("What is IV?")

    assert result.pipeline == "gemini"
    assert result.retrieved_chunk_ids == []
    assert result.response_text == "answer"
    assert result.generation_provider == "gemini"


# ---------------------------------------------------------------------------
# Baseline RAG pipeline
# ---------------------------------------------------------------------------

async def test_baseline_pipeline_returns_valid_result():
    import src.pipelines.baseline_pipeline as bp

    mock_chunks = [_text_chunk()]
    mock_gen = _gen_result()
    mock_citations = ["Test Doc (2024-11-04)"]
    mock_citation_check = {"path": "text", "verified": [], "hallucinated": []}

    with patch.object(bp, "clear_cache", new=MagicMock()), \
         patch.object(bp, "dense_search", new=AsyncMock(return_value=mock_chunks)), \
         patch.object(bp, "assemble_context", new=MagicMock(return_value=("context str", mock_citations))), \
         patch.object(bp, "generate", new=AsyncMock(return_value=mock_gen)), \
         patch.object(bp, "verify_citations", new=MagicMock(return_value=mock_citation_check)), \
         patch.object(bp, "log_result", new=MagicMock()):
        result = await bp.run("What is IV?")

    assert result.pipeline == "baseline"
    assert result.visual_path_used is False
    assert result.response_text == "mocked response"
    assert result.retrieved_chunk_ids == ["chunk-1"]


# ---------------------------------------------------------------------------
# Enhanced RAG pipeline
# ---------------------------------------------------------------------------

async def test_enhanced_pipeline_with_visual_chunk():
    import src.pipelines.enhanced_pipeline as ep
    from src.ingestion.models import ContextChunk

    text_ch = _text_chunk()
    visual_ch = _visual_chunk()
    reranked = [text_ch, visual_ch]

    mock_gen = _gen_result()
    mock_visual_gen = _gen_result(
        response_text="chart shows BTC vol",
        provider="gemini-vision",
        source_page=2,
        source_doc="Visual Doc",
    )
    mock_context_chunk = ContextChunk(
        parent_text="parent text",
        child_text="chunk text",
        metadata=text_ch.metadata,
        reranker_score=0.9,
        source_title="Test Doc",
        source_date="2024-11-04",
    )
    mock_citations = ["Test Doc (2024-11-04)"]
    mock_citation_check = {"path": "text", "verified": [], "hallucinated": []}

    with patch.object(ep, "clear_cache", new=MagicMock()), \
         patch.object(ep, "classify_query", new=AsyncMock(return_value="analytical")), \
         patch.object(ep, "rewrite_query", new=AsyncMock(return_value={"rewritten_query": "What is IV?", "sub_queries": ["What is IV?"]})), \
         patch.object(ep, "dense_search", new=AsyncMock(return_value=[text_ch])), \
         patch.object(ep, "sparse_search", new=AsyncMock(return_value=[])), \
         patch.object(ep, "rrf_merge", new=MagicMock(return_value=reranked)), \
         patch.object(ep, "rerank", new=AsyncMock(return_value=reranked)), \
         patch.object(ep, "route", new=MagicMock(return_value=([text_ch], [visual_ch]))), \
         patch.object(ep, "fetch_parents", new=AsyncMock(return_value=[mock_context_chunk])), \
         patch.object(ep, "assemble_context", new=MagicMock(return_value=("context str", mock_citations))), \
         patch.object(ep, "generate", new=AsyncMock(return_value=mock_gen)), \
         patch.object(ep, "generate_from_page", new=AsyncMock(return_value=mock_visual_gen)), \
         patch.object(ep, "verify_citations", new=MagicMock(return_value=mock_citation_check)), \
         patch.object(ep, "log_result", new=MagicMock()):
        result = await ep.run("What is IV?")

    assert result.pipeline == "enhanced"
    assert result.visual_path_used is True
    assert result.visual_pages_rendered == [2]
    assert "chart shows BTC vol" in result.response_text
    assert "mocked response" in result.response_text
