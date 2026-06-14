from __future__ import annotations

import time
import uuid

from qdrant_client.models import PointStruct

from src.config.qdrant_client import qdrant
from src.config.settings import settings
from src.embedding.embedder import embed_chunks
from src.ingestion.models import Chunk
from src.utils.logger import get_logger
from src.utils.token_counter import count_tokens

_logger = get_logger(__name__)

_UPSERT_BATCH_SIZE = 100
_UUID_NS = uuid.UUID("b10c5c01-e5c0-4e5c-b10c-5c01e5c04e5c")


def chunk_uuid(chunk_id: str) -> str:
    """Derive a deterministic UUID string from a chunk_id string."""
    return str(uuid.uuid5(_UUID_NS, chunk_id))


async def upsert_chunks(chunks: list[Chunk]) -> int:
    """Embed child chunks and upsert them into Qdrant.

    Returns the total number of points upserted.
    """
    t0 = time.perf_counter()

    # 1. Separate children and build parent lookup
    children = [c for c in chunks if not c.is_parent]
    parents: dict[str, Chunk] = {c.chunk_id: c for c in chunks if c.is_parent}

    if not children:
        return 0

    # 3. Embed all children
    id_vector_pairs = await embed_chunks(children)
    child_by_id = {c.chunk_id: c for c in children}

    # 4. Build PointStruct list
    points: list[PointStruct] = []
    total_tokens = 0

    for chunk_id, vector in id_vector_pairs:
        chunk = child_by_id[chunk_id]
        parent = parents.get(chunk.parent_id)
        total_tokens += count_tokens(chunk.text)

        payload = {
            "chunk_id":           chunk.chunk_id,
            "doc_id":             chunk.doc_id,
            "parent_id":          chunk.parent_id,
            "text":               chunk.text,
            "parent_text":        parent.text if parent else chunk.text,
            "is_parent":          False,
            "section_title":      chunk.section_title,
            "page_number":        chunk.page_number,
            "has_visual_content": chunk.has_visual_content,
            "pdf_path":           chunk.pdf_path,
            "title":              chunk.metadata.get("title", ""),
            "published_date":     chunk.metadata.get("published_date", ""),
            "doc_type":           chunk.metadata.get("doc_type", ""),
            "asset_class":        chunk.metadata.get("asset_class", ""),
            "source_file":        chunk.metadata.get("source_file", ""),
        }
        points.append(PointStruct(id=chunk_uuid(chunk_id), vector=vector, payload=payload))

    # 5. Upsert in batches of 100
    for i in range(0, len(points), _UPSERT_BATCH_SIZE):
        batch = points[i : i + _UPSERT_BATCH_SIZE]
        try:
            await qdrant.upsert(
                collection_name=settings.QDRANT_COLLECTION,
                points=batch,
            )
        except Exception as exc:
            _logger.error(
                "upsert_failed",
                extra={"batch_start": i, "batch_size": len(batch), "error": str(exc)},
            )
            raise

    elapsed_ms = (time.perf_counter() - t0) * 1_000

    # 6. Log
    _logger.info(
        "upsert_chunks",
        extra={
            "total_points": len(points),
            "total_tokens": total_tokens,
            "elapsed_ms": round(elapsed_ms, 3),
        },
    )

    return len(points)
