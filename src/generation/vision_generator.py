from __future__ import annotations

import time

from src.config.gemini_client import generate_vision
from src.config.settings import settings
from src.generation.prompts import VISION_PROMPT_TEMPLATE
from src.generation.models import GenerationResult
from src.retrieval.dense_retriever import ScoredChunk
from src.utils.logger import get_logger
from src.utils.page_renderer import PageRenderError, render_page

_logger = get_logger(__name__)


async def generate_from_page(
    chunk: ScoredChunk,
    query: str,
) -> GenerationResult:
    """Generate an answer from the rendered PDF page that contains *chunk*.

    Args:
        chunk: A visual ScoredChunk with pdf_path and page_number in metadata.
        query: The user's question.

    Returns:
        GenerationResult with response_text, provider="gemini-vision",
        source_doc, and source_page populated.
    """
    t0 = time.perf_counter()

    pdf_path = chunk.metadata.get("pdf_path", "")
    page_number = chunk.metadata.get("page_number", 1)
    source_doc = chunk.metadata.get("title", "Unknown")
    source_date = chunk.metadata.get("published_date", "")

    # 2. Render the page
    try:
        pil_image = render_page(pdf_path, page_number)
    except PageRenderError as e:
        _logger.warning(
            "vision_render_failed",
            extra={"pdf_path": pdf_path, "page_number": page_number, "error": str(e)},
        )
        return GenerationResult(
            response_text="[Visual content unavailable for this page]",
            provider="gemini-vision",
            source_doc=source_doc,
            source_page=page_number,
        )

    # 3. Build vision prompt
    prompt_text = VISION_PROMPT_TEMPLATE.format(
        query=query,
        source_doc=source_doc,
        source_page=page_number,
    )

    # 4. Call vision model
    response_text = await generate_vision(pil_image, prompt_text)

    elapsed_ms = (time.perf_counter() - t0) * 1_000

    # 6. Log
    _logger.info(
        "generate_from_page",
        extra={
            "pdf_path": pdf_path,
            "page_number": page_number,
            "source_doc": source_doc,
            "elapsed_ms": round(elapsed_ms, 3),
        },
    )

    # 5. Return
    return GenerationResult(
        response_text=response_text,
        provider="gemini-vision",
        model_name=settings.GEMINI_VISION_MODEL,
        source_page=page_number,
        source_doc=source_doc,
        latency_ms=elapsed_ms,
    )
