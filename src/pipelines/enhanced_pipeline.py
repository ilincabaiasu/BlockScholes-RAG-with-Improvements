from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

from src.ablation.config import FULL, AblationConfig
from src.config.settings import settings
from src.generation.context_assembler import assemble_context
from src.generation.citation_verifier import verify_citations
from src.generation.generator import generate
from src.generation.vision_generator import generate_from_page
from src.pipelines.models import GenerationResult, QueryResult
from src.pipelines.query_logger import log_result
from src.query_processing.classifier import classify_query
from src.query_processing.rewriter import rewrite_query
from src.query_processing.scope_detector import detect_scope, extract_target_period, scope_params
from src.retrieval.dense_retriever import ScoredChunk, dense_search
from src.retrieval.hybrid_merger import rrf_merge
from src.retrieval.parent_fetcher import fetch_parents
from src.retrieval.reranker import rerank
from src.retrieval.sparse_retriever import sparse_search
from src.retrieval.visual_router import route
from src.utils.logger import get_logger
from src.utils.page_renderer import clear_cache
from src.utils.timer import Timer

_logger = get_logger(__name__)


async def run(
    query: str,
    filters: dict | None = None,
    config: AblationConfig | None = None,
) -> QueryResult:
    """Run the full enhanced RAG pipeline.

    Includes query classification, scope detection, rewriting, hybrid retrieval
    (dense + BM25), RRF merging, cross-encoder reranking, visual routing,
    parent-chunk fetching, context assembly, and generation.

    Retrieval breadth (context_top_k, reranker_top_k, max_context_tokens) scales
    automatically with the detected temporal scope of the query so that month-wide
    questions retrieve more sources than single-day lookups.

    Args:
        query:   The user's question.
        filters: Optional Qdrant payload filters (doc_type, asset_class).
        config:  Ablation flags. Defaults to FULL (every enhancement on), which
                 reproduces the production behaviour. Passing a config with flags
                 disabled is used by the ablation harness.

    Returns:
        A fully populated QueryResult logged to data/query_log.jsonl.
    """
    if config is None:
        config = FULL

    # 1. Initialise
    clear_cache()
    query_id = str(uuid.uuid4())[:8]
    latency: dict = {}

    # 2. Classify + detect temporal scope (both pure/cheap, run first)
    with Timer("classify") as t:
        # Ablating classification falls back to "factual_lookup" + the vanilla
        # baseline prompt, so a fully-disabled config reproduces the baseline
        # pipeline's generation exactly (temp 0.0, BASELINE_SYSTEM_PROMPT).
        query_type = await classify_query(query) if config.classify else "factual_lookup"
    latency["classify_ms"] = t.elapsed_ms

    # Classification also selects the system prompt: enhanced when on, the
    # vanilla baseline prompt when ablated.
    prompt_variant = "enhanced" if config.classify else "baseline"

    scope = detect_scope(query)
    target_period = extract_target_period(query)  # e.g. "2026-03" or None
    # Scope-adaptive breadth, or static settings defaults when ablated.
    if config.scope_adaptive:
        params = scope_params(scope)
        ctx_top_k = params["context_top_k"]
        reranker_top_k = params["reranker_top_k"]
        max_ctx_tokens = params["max_context_tokens"]
    else:
        ctx_top_k = settings.CONTEXT_TOP_K
        reranker_top_k = settings.RERANKER_TOP_K
        max_ctx_tokens = settings.MAX_CONTEXT_TOKENS

    _logger.info(
        "scope_params_applied",
        extra={
            "scope": scope,
            "context_top_k": ctx_top_k,
            "reranker_top_k": reranker_top_k,
            "max_context_tokens": max_ctx_tokens,
        },
    )

    # 3. Rewrite — ablated to the original query only (no sub-query decomposition)
    with Timer("rewrite") as t:
        if config.rewrite:
            rewrite_result = await rewrite_query(query)
            rewritten = rewrite_result["rewritten_query"]
            sub_queries = rewrite_result["sub_queries"]
        else:
            rewritten = query
            sub_queries = [query]
    latency["rewrite_ms"] = t.elapsed_ms

    # 4. Hybrid retrieval across all sub-queries. With hybrid ablated, only the
    # dense arm runs and the sparse list stays empty (RRF then degenerates to
    # dense-only ranking).
    with Timer("retrieve") as t:
        all_dense: list[ScoredChunk] = []
        all_sparse: list[ScoredChunk] = []

        for sq in sub_queries:
            if config.hybrid:
                d_res, s_res = await asyncio.gather(
                    dense_search(sq, top_k=settings.DENSE_TOP_K, filters=filters),
                    sparse_search(sq, top_k=settings.SPARSE_TOP_K),
                )
            else:
                d_res = await dense_search(sq, top_k=settings.DENSE_TOP_K, filters=filters)
                s_res = []
            all_dense.extend(d_res)
            all_sparse.extend(s_res)

        # Deduplicate dense — keep highest dense_score per chunk_id
        seen_dense: dict[str, ScoredChunk] = {}
        for c in all_dense:
            if c.chunk_id not in seen_dense or c.dense_score > seen_dense[c.chunk_id].dense_score:
                seen_dense[c.chunk_id] = c
        all_dense = list(seen_dense.values())

        # Deduplicate sparse — keep highest sparse_score per chunk_id
        seen_sparse: dict[str, ScoredChunk] = {}
        for c in all_sparse:
            if c.chunk_id not in seen_sparse or c.sparse_score > seen_sparse[c.chunk_id].sparse_score:
                seen_sparse[c.chunk_id] = c
        all_sparse = list(seen_sparse.values())

    latency["retrieve_ms"] = t.elapsed_ms

    # 5. RRF merge
    merged = rrf_merge(all_dense, all_sparse)

    # 6. Rerank — fall back to RRF-sorted merged list if reranker returns empty.
    # Ablated: skip the cross-encoder and keep the RRF/dense ordering.
    with Timer("rerank") as t:
        if config.rerank:
            try:
                reranked = await rerank(rewritten, merged, top_k=reranker_top_k)
            except Exception as exc:
                _logger.warning("rerank_failed_fallback", extra={"error": str(exc)})
                reranked = []
            if not reranked:
                _logger.warning("rerank_empty_fallback", extra={"merged_count": len(merged)})
                reranked = merged[:reranker_top_k]
        else:
            reranked = merged[:reranker_top_k]
    latency["rerank_ms"] = t.elapsed_ms

    # 6b. Temporal re-prioritisation — when the query targets a specific period
    # (e.g. "March 2026"), chunks from that period are moved to the front so
    # they are not displaced by higher-scoring but out-of-period articles.
    if target_period and config.temporal_reprioritize:
        in_period = [
            c for c in reranked
            if target_period in c.metadata.get("published_date", "")
        ]
        out_of_period = [
            c for c in reranked
            if target_period not in c.metadata.get("published_date", "")
        ]
        reranked = in_period + out_of_period
        _logger.info(
            "temporal_reprioritise",
            extra={
                "target_period": target_period,
                "in_period": len(in_period),
                "out_of_period": len(out_of_period),
            },
        )

    # 6c. Source diversity — cap chunks per document so no single article
    # dominates the context window. Preserves reranker score ordering.
    if config.diversity_cap:
        _doc_counts: dict[str, int] = {}
        diverse: list[ScoredChunk] = []
        for chunk in reranked:
            doc_id = chunk.metadata.get("doc_id", chunk.chunk_id)
            if _doc_counts.get(doc_id, 0) < settings.MAX_CHUNKS_PER_DOC:
                diverse.append(chunk)
                _doc_counts[doc_id] = _doc_counts.get(doc_id, 0) + 1
        reranked = diverse

    # 7. Visual routing — only for query types that benefit from charts.
    # Definitional queries (e.g. "What is IV?") are answered from text alone:
    # they need a concept, not chart values, and the text path already carries
    # the visual chunks' text. Ablated (config.visual off): route nothing.
    with Timer("route") as t:
        if config.visual and query_type != "definitional":
            text_chunks, visual_chunks = route(reranked)
        else:
            text_chunks, visual_chunks = reranked, []
    latency["route_ms"] = t.elapsed_ms

    # 8. Text and visual paths in parallel
    # Text path uses ALL top reranked chunks (text + visual) so that
    # visual chunk text content is never lost when rendering fails.
    async def text_path() -> tuple[GenerationResult, str, list[str]]:
        context_chunks = await fetch_parents(
            reranked[:ctx_top_k],
            max_context_tokens=max_ctx_tokens,
            use_parent=config.parent_fetch,
        )
        context_str, citations = assemble_context(context_chunks, max_context_tokens=max_ctx_tokens)
        gen = await generate(context_str, query, query_type, prompt_variant)
        return gen, context_str, citations

    async def visual_path() -> tuple[list[GenerationResult], list[int]]:
        results: list[GenerationResult] = []
        pages_rendered: list[int] = []
        seen_pages: set[tuple[str, int]] = set()
        for vc in visual_chunks:
            pdf_path = vc.metadata.get("pdf_path", "")
            page_number = vc.metadata.get("page_number", 1)
            key = (pdf_path, page_number)
            if key in seen_pages:
                continue
            seen_pages.add(key)
            try:
                vr = await generate_from_page(vc, query)
            except Exception as exc:
                _logger.warning(
                    "visual_path_chunk_failed",
                    extra={"pdf_path": pdf_path, "page_number": page_number, "error": str(exc)},
                )
                continue
            results.append(vr)
            pages_rendered.append(vr.source_page)
            if len(results) >= 2:  # cap at 2 unique pages
                break
        return results, pages_rendered

    with Timer("generate") as t:
        (text_gen, context_str, citations), (visual_results, pages) = (
            await asyncio.gather(text_path(), visual_path())
        )
    latency["text_generate_ms"] = t.elapsed_ms

    # 9. Merge text and visual outputs
    # Filter out unavailable visual results (poppler not installed etc.)
    available_visual = [
        vr for vr in visual_results
        if "[Visual content unavailable" not in vr.response_text
    ]

    final_text = ""
    if available_visual:
        for vr in available_visual:
            final_text += (
                f"[Visual source — Page {vr.source_page}, "
                f"{vr.source_doc}]:\n{vr.response_text}\n\n"
            )
    if text_gen.response_text:
        final_text += text_gen.response_text
    final_text = final_text.strip()

    # 10. Citation verification
    citation_check = verify_citations(text_gen.response_text, citations)
    visual_cit = ({"path": "visual", "verified": True} if available_visual else {})
    combined_citations = {**citation_check, **visual_cit}

    final_gen, was_refined = text_gen, False

    # 12. Build result
    result = QueryResult(
        query_id=query_id,
        original_query=query,
        rewritten_query=rewritten,
        query_type=query_type,
        pipeline="enhanced",
        retrieved_chunk_ids=[c.chunk_id for c in reranked],
        reranker_scores=[c.reranker_score for c in reranked],
        context_sources=citations,
        response_text=final_text,
        refined=was_refined,
        citation_verification=combined_citations,
        generation_provider=text_gen.provider,
        latency_breakdown=latency,
        visual_path_used=len(available_visual) > 0,
        visual_pages_rendered=[vr.source_page for vr in available_visual],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # 13. Log and return
    log_result(result)
    return result
