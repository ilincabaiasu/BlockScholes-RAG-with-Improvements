from __future__ import annotations

from qdrant_client.models import Distance, HnswConfigDiff, PayloadSchemaType, VectorParams

from src.config.qdrant_client import qdrant
from src.config.settings import settings
from src.utils.logger import get_logger

_logger = get_logger(__name__)

_PAYLOAD_INDEXES: list[tuple[str, PayloadSchemaType]] = [
    ("doc_type", PayloadSchemaType.KEYWORD),
    ("asset_class", PayloadSchemaType.KEYWORD),
    ("is_parent", PayloadSchemaType.BOOL),
    ("has_visual_content", PayloadSchemaType.BOOL),
    ("page_number", PayloadSchemaType.INTEGER),
    ("published_date", PayloadSchemaType.KEYWORD),
]


async def create_collection() -> None:
    """Create the Qdrant collection and payload indexes if they don't exist."""
    if await qdrant.collection_exists(settings.QDRANT_COLLECTION):
        _logger.info(
            "collection_already_exists",
            extra={"collection": settings.QDRANT_COLLECTION},
        )
        return

    await qdrant.create_collection(
        collection_name=settings.QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=settings.EMBEDDING_DIM,
            distance=Distance.COSINE,
        ),
        hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
    )

    for field_name, schema_type in _PAYLOAD_INDEXES:
        await qdrant.create_payload_index(
            collection_name=settings.QDRANT_COLLECTION,
            field_name=field_name,
            field_schema=schema_type,
        )

    _logger.info(
        "collection_created",
        extra={
            "collection": settings.QDRANT_COLLECTION,
            "dim": settings.EMBEDDING_DIM,
            "indexes": [f for f, _ in _PAYLOAD_INDEXES],
        },
    )


async def delete_collection() -> None:
    """Delete the Qdrant collection. Used during development for full re-index."""
    await qdrant.delete_collection(settings.QDRANT_COLLECTION)
    _logger.info(
        "collection_deleted",
        extra={"collection": settings.QDRANT_COLLECTION},
    )
