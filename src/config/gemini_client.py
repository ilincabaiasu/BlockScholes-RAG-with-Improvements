# Allowed importers — do not import this module from anywhere else:
#   src/embedding/embedder.py
#   src/generation/generator.py
#   src/generation/vision_generator.py
#   src/pipelines/gemini_pipeline.py
#   src/query_processing/classifier.py
#   src/query_processing/rewriter.py

from __future__ import annotations

from google import genai
from google.genai import types
from google.genai.errors import ClientError

from src.config.settings import settings
from src.utils.logger import get_logger
from src.utils.retry import with_exponential_backoff
from src.utils.timer import Timer

_logger = get_logger(__name__)

# Module-level client — shared across both generation functions.
gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)

# Convenience aliases so callers can reference the model names if needed.
gemini_text = settings.GEMINI_MODEL
gemini_vision = settings.GEMINI_VISION_MODEL

# Gemini rate-limit exception predicate — ClientError is broader (all 4xx),
# so we subclass to narrow it to HTTP 429 only.
class _RateLimitError(Exception):
    pass


def _wrap_client_error(exc: ClientError) -> BaseException:
    """Re-raise ClientError as _RateLimitError for HTTP 429 and 503."""
    if exc.code in (429, 503):
        raise _RateLimitError(str(exc)) from exc
    raise exc


async def generate_text(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.0,
) -> str:
    """Generate a text response from Gemini.

    Combines *system_prompt* and *user_message* into a single prompt string
    and calls the Gemini text model. Retries up to 4 times on HTTP 429
    with exponential backoff starting at 1 s.
    """
    prompt = f"{system_prompt}\n\n{user_message}"
    config = types.GenerateContentConfig(temperature=temperature)

    async def _call() -> str:
        try:
            with Timer("text_generate") as t:
                response = await gemini_client.aio.models.generate_content(
                    model=settings.GEMINI_MODEL,
                    contents=prompt,
                    config=config,
                )
        except ClientError as exc:
            _wrap_client_error(exc)
        _logger.info(
            "generate_text",
            extra={
                "function": "generate_text",
                "model": settings.GEMINI_MODEL,
                "elapsed_ms": t.elapsed_ms,
                "call_type": "text",
            },
        )
        if not response.text:
            raise ValueError("Gemini returned an empty response (safety filter or empty completion).")
        return response.text

    return await with_exponential_backoff(
        _call,
        retry_on=_RateLimitError,
        label="gemini.text",
    )


async def generate_vision(
    pil_image: object,  # PIL.Image.Image — avoid hard PIL import at module level
    prompt_text: str,
    temperature: float = 0.0,
) -> str:
    """Generate a response from Gemini given a PIL image and a text prompt.

    Passes [pil_image, prompt_text] as the content list to the Gemini vision
    model. Retries up to 4 times on HTTP 429 with exponential backoff.
    """
    content = [pil_image, prompt_text]
    config = types.GenerateContentConfig(temperature=temperature)

    async def _call() -> str:
        try:
            with Timer("vision_generate") as t:
                response = await gemini_client.aio.models.generate_content(
                    model=settings.GEMINI_VISION_MODEL,
                    contents=content,
                    config=config,
                )
        except ClientError as exc:
            _wrap_client_error(exc)
        _logger.info(
            "generate_vision",
            extra={
                "function": "generate_vision",
                "model": settings.GEMINI_VISION_MODEL,
                "elapsed_ms": t.elapsed_ms,
                "call_type": "vision",
            },
        )
        if not response.text:
            raise ValueError("Gemini vision returned an empty response (safety filter or empty completion).")
        return response.text

    return await with_exponential_backoff(
        _call,
        retry_on=_RateLimitError,
        label="gemini.vision",
    )


_EMBED_BATCH_SIZE = 20   # smaller batches → fewer tokens per request
_EMBED_BATCH_DELAY_S = 1.0  # pause between batches to avoid rate limits


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed *texts* using Gemini gemini-embedding-001 and return float vectors.

    Texts are sent in batches of 20 with a 1 s pause between batches.
    Retries on HTTP 429 with exponential backoff.
    Vectors are returned in the same order as the input.

    Raises:
        ValueError: if *texts* is empty.
    """
    import asyncio

    if not texts:
        raise ValueError("embed_texts requires at least one input string.")

    batches = [
        texts[i : i + _EMBED_BATCH_SIZE]
        for i in range(0, len(texts), _EMBED_BATCH_SIZE)
    ]
    vectors: list[list[float]] = []

    for batch_idx, batch in enumerate(batches):
        # Pause between batches (skip before the first one)
        if batch_idx > 0:
            await asyncio.sleep(_EMBED_BATCH_DELAY_S)

        async def _call(b: list[str] = batch) -> list[list[float]]:
            try:
                with Timer("embed_batch") as t:
                    response = await gemini_client.aio.models.embed_content(
                        model=settings.EMBEDDING_MODEL,
                        contents=b,
                    )
            except ClientError as exc:
                _wrap_client_error(exc)
            _logger.info(
                "embed_texts",
                extra={
                    "model": settings.EMBEDDING_MODEL,
                    "input_count": len(b),
                    "elapsed_ms": t.elapsed_ms,
                },
            )
            return [e.values for e in response.embeddings]

        batch_result = await with_exponential_backoff(
            _call,
            retry_on=_RateLimitError,
            label="gemini.embed",
        )
        vectors.extend(batch_result)

    return vectors
