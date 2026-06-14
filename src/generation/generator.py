from __future__ import annotations

import time

from src.config.gemini_client import generate_text
from src.config.openai_client import oai_client
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

    # 4. Route by provider
    provider = settings.GENERATION_PROVIDER

    if provider == "gemini":
        response_text = await generate_text(system_prompt, user_message, temperature)

    elif provider == "openai":
        response = await oai_client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=settings.MAX_COMPLETION_TOKENS,
        )
        response_text = response.choices[0].message.content

    else:
        raise ValueError(f"Unknown GENERATION_PROVIDER: {provider!r}")

    elapsed_ms = (time.perf_counter() - t0) * 1_000

    # 5. Log
    _logger.info(
        "generate",
        extra={
            "provider": provider,
            "query_type": query_type,
            "prompt_variant": prompt_variant,
            "elapsed_ms": round(elapsed_ms, 3),
        },
    )

    # 6. Return
    model_name = settings.GEMINI_MODEL if provider == "gemini" else settings.OPENAI_MODEL
    return GenerationResult(
        response_text=response_text,
        provider=provider,
        model_name=model_name,
        latency_ms=elapsed_ms,
    )
