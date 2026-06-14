from __future__ import annotations

import asyncio

import openai

from src.config.settings import settings
from src.utils.logger import get_logger
from src.utils.retry import with_exponential_backoff
from src.utils.timer import Timer

_logger = get_logger(__name__)

oai_client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

_BATCH_SIZE = 100


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed *texts* using OpenAI and return float vectors in input order.

    Texts are sent in batches of 100. RateLimitError triggers exponential
    backoff (1 s → 2 s → 4 s) up to 4 attempts per batch.

    Raises:
        ValueError: if *texts* is empty.
    """
    if not texts:
        raise ValueError("embed_texts requires at least one input string.")

    batches = [texts[i : i + _BATCH_SIZE] for i in range(0, len(texts), _BATCH_SIZE)]
    vectors: list[list[float]] = []

    for batch in batches:
        async def _call() -> list[list[float]]:
            with Timer("embed_batch") as t:
                response = await oai_client.embeddings.create(
                    model=settings.EMBEDDING_MODEL,
                    input=batch,
                )
            _logger.info(
                "embed_texts",
                extra={
                    "model": settings.EMBEDDING_MODEL,
                    "input_count": len(batch),
                    "elapsed_ms": t.elapsed_ms,
                },
            )
            batch_vectors: list[list[float]] = [None] * len(batch)  # type: ignore[list-item]
            for item in response.data:
                batch_vectors[item.index] = item.embedding
            return batch_vectors

        batch_result = await with_exponential_backoff(
            _call,
            retry_on=openai.RateLimitError,
            label="openai.embed",
        )
        vectors.extend(batch_result)

    return vectors
