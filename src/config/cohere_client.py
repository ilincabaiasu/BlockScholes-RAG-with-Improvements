# Imported only by:
#   src/retrieval/reranker.py

from __future__ import annotations

import cohere
import cohere.errors

from src.config.settings import settings
from src.utils.logger import get_logger
from src.utils.retry import with_exponential_backoff

_logger = get_logger(__name__)

co_client = cohere.AsyncClient(api_key=settings.COHERE_API_KEY)


async def rerank_with_retry(
    query: str,
    documents: list[str],
    top_n: int,
    model: str,
) -> cohere.RerankResponse:
    """Call co_client.rerank() with exponential backoff on TooManyRequestsError.

    Retries up to 4 attempts, starting at 1 s and doubling each time.
    """
    return await with_exponential_backoff(
        lambda: co_client.rerank(
            query=query,
            documents=documents,
            top_n=top_n,
            model=model,
        ),
        retry_on=cohere.errors.TooManyRequestsError,
        label="cohere.rerank",
    )
