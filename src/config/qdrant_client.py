# Imported by:
#   src/indexing/collection_manager.py
#   src/indexing/upserter.py
#   src/retrieval/dense_retriever.py
#   src/retrieval/parent_fetcher.py

from __future__ import annotations

from qdrant_client import AsyncQdrantClient

from src.config.settings import settings
from src.utils.logger import get_logger

_logger = get_logger(__name__)

qdrant = AsyncQdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)


async def verify_connection() -> None:
    """Check that Qdrant is reachable and log the available collections.

    Raises:
        RuntimeError: if the server cannot be reached.
    """
    try:
        result = await qdrant.get_collections()
        collection_names = [c.name for c in result.collections]
        _logger.info(
            "qdrant_connected",
            extra={"collections": collection_names},
        )
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach Qdrant at {settings.QDRANT_URL} — is the server running?"
        ) from exc
