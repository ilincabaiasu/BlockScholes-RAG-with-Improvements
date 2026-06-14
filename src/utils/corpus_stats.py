from __future__ import annotations

from src.config.qdrant_client import qdrant
from src.config.settings import settings
from src.utils.logger import get_logger

_logger = get_logger(__name__)


async def get_corpus_stats() -> dict:
    """Scroll the Qdrant collection and return corpus-level metadata.

    Collects one record per unique doc_id (ignores duplicate chunk payloads)
    so the numbers reflect documents, not chunks.

    Returns a dict with keys:
        doc_count     – total unique documents indexed
        date_min      – earliest published_date string (YYYY-MM-DD)
        date_max      – latest  published_date string (YYYY-MM-DD)
        recent        – list of (title, date) for the 6 most recent docs
    """
    seen: dict[str, dict] = {}  # doc_id → {title, date}
    offset = None

    while True:
        results, next_offset = await qdrant.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            limit=250,
            offset=offset,
            with_payload=["doc_id", "title", "published_date"],
            with_vectors=False,
        )

        for point in results:
            p = point.payload or {}
            doc_id = p.get("doc_id", "")
            if doc_id and doc_id not in seen:
                seen[doc_id] = {
                    "title": p.get("title", ""),
                    "date":  p.get("published_date", ""),
                }

        if next_offset is None:
            break
        offset = next_offset

    if not seen:
        return {}

    docs  = list(seen.values())
    dates = sorted(d["date"] for d in docs if d["date"])
    by_date = sorted(docs, key=lambda d: d["date"], reverse=True)

    _logger.info(
        "corpus_stats",
        extra={
            "doc_count": len(docs),
            "date_min": dates[0] if dates else "",
            "date_max": dates[-1] if dates else "",
        },
    )

    return {
        "doc_count": len(docs),
        "date_min":  dates[0]  if dates else "—",
        "date_max":  dates[-1] if dates else "—",
        "recent":    [(d["title"], d["date"]) for d in by_date[:6]],
    }
