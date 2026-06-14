from __future__ import annotations

import time

from src.config.gemini_client import embed_texts
from src.config.settings import settings
from src.utils.logger import get_logger
from src.utils.token_counter import count_tokens

_logger = get_logger(__name__)


class EmbeddingError(Exception):
    pass


async def embed_chunks(chunks: list) -> list[tuple[str, list[float]]]:
    """Embed a list of Chunk objects and return (chunk_id, vector) tuples.

    Raises:
        EmbeddingError: if any vector is all zeros or has wrong dimension.
    """
    texts = [c.text for c in chunks]
    chunk_ids = [c.chunk_id for c in chunks]
    total_tokens = sum(count_tokens(t) for t in texts)

    t0 = time.perf_counter()
    vectors = await embed_texts(texts)
    elapsed_ms = (time.perf_counter() - t0) * 1_000

    results: list[tuple[str, list[float]]] = []
    for chunk_id, vector in zip(chunk_ids, vectors):
        if len(vector) != settings.EMBEDDING_DIM:
            raise EmbeddingError(
                f"Vector for {chunk_id} has dimension {len(vector)}, "
                f"expected {settings.EMBEDDING_DIM}"
            )
        if all(v == 0.0 for v in vector):
            raise EmbeddingError(f"Vector for {chunk_id} is all zeros")
        results.append((chunk_id, vector))

    _logger.info(
        "embed_chunks",
        extra={
            "total_chunks": len(chunks),
            "total_tokens": total_tokens,
            "elapsed_ms": round(elapsed_ms, 3),
        },
    )

    return results


async def embed_query(query_text: str) -> list[float]:
    """Embed a single query string and return its float vector.

    Raises:
        ValueError: if query_text is empty.
    """
    if not query_text:
        raise ValueError("query_text must not be empty")

    vectors = await embed_texts([query_text])
    return vectors[0]
