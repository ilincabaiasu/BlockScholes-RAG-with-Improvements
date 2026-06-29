from __future__ import annotations

import uuid
from datetime import datetime, timezone

from src.config.gemini_client import generate_text
from src.config.settings import settings
from src.generation.prompts import GEMINI_VANILLA_SYSTEM_PROMPT
from src.pipelines.models import QueryResult
from src.pipelines.query_logger import log_result
from src.utils.logger import get_logger
from src.utils.page_renderer import clear_cache
from src.utils.timer import Timer

_logger = get_logger(__name__)


async def run(query: str) -> QueryResult:
    """Run the Gemini vanilla (no-RAG) pipeline.

    Sends *query* directly to Gemini with zero retrieved context.
    Used as a comparison baseline against the RAG pipelines.

    Args:
        query: The user's question.

    Returns:
        A fully populated QueryResult logged to data/query_log.jsonl.
    """
    clear_cache()
    query_id = str(uuid.uuid4())[:8]
    latency: dict = {}

    with Timer("text_generate") as t:
        response_text = await generate_text(
            GEMINI_VANILLA_SYSTEM_PROMPT, query, temperature=0.0
        )
    latency["text_generate_ms"] = t.elapsed_ms

    result = QueryResult(
        query_id=query_id,
        original_query=query,
        pipeline="gemini",
        response_text=response_text,
        generation_provider="gemini",
        latency_breakdown=latency,
        timestamp=datetime.now(timezone.utc).isoformat(),
        retrieved_chunk_ids=[],
        context_sources=[],
        reranker_scores=[],
        visual_path_used=False,
        visual_pages_rendered=[],
    )

    log_result(result)
    return result
