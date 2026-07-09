from __future__ import annotations

import time

from src.config.openai_client import generate_text as _generate_text_openai
from src.config.settings import settings
from src.generation.prompts import BASELINE_SYSTEM_PROMPT, ENHANCED_SYSTEM_PROMPT
from src.generation.models import GenerationResult
from src.utils.logger import get_logger

_logger = get_logger(__name__)

_FACTUAL_TYPES = frozenset({"factual_lookup", "definitional"})


async def generate(
    context: str,
    query: str,
    query_type: str,
    prompt_variant: str = "enhanced",  # "baseline" | "enhanced"
) -> GenerationResult:
    """Generate an answer from *context* and *query*.

    Args:
        context:        Pre-assembled context string (text + optional vision).
        query:          The user's original question.
        query_type:     One of factual_lookup | definitional | analytical | comparative.
        prompt_variant: "baseline" or "enhanced" — selects the system prompt.

    Returns:
        GenerationResult with response_text, provider, and model_name populated.
    """
    t0 = time.perf_counter()

    # 1. Select temperature
    temperature = (
        settings.TEMP_FACTUAL
        if query_type in _FACTUAL_TYPES
        else settings.TEMP_ANALYTICAL
    )

    # 2. Select system prompt
    system_prompt = (
        BASELINE_SYSTEM_PROMPT
        if prompt_variant == "baseline"
        else ENHANCED_SYSTEM_PROMPT
    )

    # 3. Build user message
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    # 4. Generate
    response_text = await _generate_text_openai(system_prompt, user_message, temperature)

    elapsed_ms = (time.perf_counter() - t0) * 1_000

    # 5. Log
    _logger.info(
        "generate",
        extra={
            "provider": "openai",
            "query_type": query_type,
            "prompt_variant": prompt_variant,
            "elapsed_ms": round(elapsed_ms, 3),
        },
    )

    # 6. Return
    return GenerationResult(
        response_text=response_text,
        provider="openai",
        model_name=settings.OPENAI_MODEL,
        latency_ms=elapsed_ms,
    )
