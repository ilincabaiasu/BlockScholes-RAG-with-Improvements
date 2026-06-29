from __future__ import annotations

import uuid
from datetime import datetime, timezone

from src.config.settings import settings
from src.generation.context_assembler import assemble_context
from src.generation.citation_verifier import verify_citations
from src.generation.generator import generate
from src.ingestion.models import ContextChunk
from src.pipelines.models import QueryResult
from src.pipelines.query_logger import log_result
from src.retrieval.dense_retriever import dense_search
from src.utils.logger import get_logger
from src.utils.page_renderer import clear_cache
from src.utils.timer import Timer

_logger = get_logger(__name__)


async def run(query: str, filters: dict | None = None) -> QueryResult:
    """Run the baseline RAG pipeline.

    Dense-only retrieval with fixed-size chunks and a vanilla prompt.
    No query rewriting, BM25, reranking, visual path, or refinement.

    Args:
        query:   The user's question.
        filters: Optional Qdrant payload filters (doc_type, asset_class).

    Returns:
        A fully populated QueryResult logged to data/query_log.jsonl.
    """
    # 1. Reset page render cache
    clear_cache()

    # 2. Identifiers
    query_id = str(uuid.uuid4())[:8]
    latency: dict = {}

    # 3. Dense retrieval
    with Timer("retrieve") as t:
        chunks = await dense_search(
            query,
            top_k=settings.CONTEXT_TOP_K,
            filters=filters,
        )
    latency["retrieve_ms"] = t.elapsed_ms

    # 4. Convert ScoredChunk → ContextChunk
    context_chunks: list[ContextChunk] = [
        ContextChunk(
            parent_text=chunk.metadata.get("parent_text", chunk.text),
            child_text=chunk.text,
            metadata=chunk.metadata,
            reranker_score=chunk.dense_score,
            source_title=chunk.metadata.get("title", ""),
            source_date=chunk.metadata.get("published_date", ""),
        )
        for chunk in chunks
    ]

    # 5. Assemble context and generate
    with Timer("text_generate") as t:
        context_str, citations = assemble_context(context_chunks)
        gen_result = await generate(
            context_str,
            query,
            query_type="factual_lookup",
            prompt_variant="baseline",
        )
    latency["text_generate_ms"] = t.elapsed_ms

    # 6. Citation verification
    citation_check = verify_citations(gen_result.response_text, citations)

    # 7. Build result
    result = QueryResult(
        query_id=query_id,
        original_query=query,
        pipeline="baseline",
        retrieved_chunk_ids=[c.chunk_id for c in chunks],
        reranker_scores=[c.dense_score for c in chunks],
        context_sources=citations,
        response_text=gen_result.response_text,
        citation_verification=citation_check,
        generation_provider=gen_result.provider,
        latency_breakdown=latency,
        visual_path_used=False,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # 8. Log and return
    log_result(result)
    return result
