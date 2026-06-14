from __future__ import annotations

import json
import time
from pathlib import Path

from src.chunking.chunker import chunk_document
from src.indexing.bm25_index import build_index
from src.indexing.collection_manager import create_collection
from src.indexing.upserter import upsert_chunks
from src.utils.logger import get_logger

_logger = get_logger(__name__)

_PROCESSED_DIR = Path("data/processed")
_MANIFEST_PATH = _PROCESSED_DIR / "index_manifest.json"
_SKIP_FILES = {"doc_hashes.json", "bm25_chunk_ids.json", "index_manifest.json"}


def _load_manifest() -> dict:
    if not _MANIFEST_PATH.is_file():
        return {}
    try:
        return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_manifest(manifest: dict) -> None:
    _MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _load_all_docs() -> list[dict]:
    """Load all ParsedDocument dicts from data/processed/, skipping index files."""
    docs = []
    for path in sorted(_PROCESSED_DIR.glob("*.json")):
        if path.name in _SKIP_FILES or path.name.endswith("_baseline_chunks.json"):
            continue
        try:
            docs.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as exc:
            _logger.warning(
                "doc_load_failed",
                extra={"path": str(path), "error": str(exc)},
            )
    return docs


def _save_baseline_sidecar(doc_id: str, baseline_chunks: list) -> None:
    path = _PROCESSED_DIR / f"{doc_id}_baseline_chunks.json"
    data = [c.model_dump() for c in baseline_chunks]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


async def _process_docs(
    docs: list[dict],
    manifest: dict,
) -> tuple[list[str], list[str], list]:
    """Process a list of docs: chunk, upsert, update manifest.

    Returns (indexed_ids, skipped_ids, all_child_chunks).
    """
    indexed: list[str] = []
    skipped: list[str] = []
    all_child_chunks: list = []

    for doc in docs:
        doc_id = doc["doc_id"]

        if doc_id in manifest:
            skipped.append(doc_id)
            # Still collect child chunks for BM25 rebuild
            enhanced = chunk_document(doc, "enhanced")
            all_child_chunks.extend(c for c in enhanced if not c.is_parent)
            continue

        # Enhanced chunks → Qdrant
        enhanced = chunk_document(doc, "enhanced")
        child_chunks = [c for c in enhanced if not c.is_parent]
        all_child_chunks.extend(child_chunks)

        # Baseline chunks → sidecar JSON
        baseline = chunk_document(doc, "baseline")
        _save_baseline_sidecar(doc_id, baseline)

        await upsert_chunks(enhanced)

        manifest[doc_id] = True
        _save_manifest(manifest)
        indexed.append(doc_id)

        _logger.info(
            "doc_indexed",
            extra={
                "doc_id": doc_id,
                "title": doc.get("title", ""),
                "enhanced_children": len(child_chunks),
                "baseline_chunks": len(baseline),
            },
        )

    return indexed, skipped, all_child_chunks


async def index_all() -> dict:
    """Ingest and index every document in data/processed/.

    Returns {"indexed": [doc_ids], "skipped": [doc_ids]}.
    """
    t0 = time.perf_counter()

    await create_collection()
    manifest = _load_manifest()
    docs = _load_all_docs()

    indexed, skipped, all_child_chunks = await _process_docs(docs, manifest)

    if all_child_chunks:
        build_index(all_child_chunks)

    elapsed_ms = (time.perf_counter() - t0) * 1_000
    _logger.info(
        "index_all_complete",
        extra={
            "indexed": len(indexed),
            "skipped": len(skipped),
            "bm25_chunks": len(all_child_chunks),
            "elapsed_ms": round(elapsed_ms, 3),
        },
    )

    return {"indexed": indexed, "skipped": skipped}


async def index_new(doc_ids: list[str]) -> dict:
    """Index only the specified doc_ids (incremental ingestion).

    BM25 is rebuilt from all known docs so existing entries are preserved.
    Returns {"indexed": [doc_ids], "skipped": [doc_ids]}.
    """
    t0 = time.perf_counter()

    await create_collection()
    manifest = _load_manifest()
    all_docs = _load_all_docs()

    # Process only the requested docs
    doc_id_set = set(doc_ids)
    target_docs = [d for d in all_docs if d["doc_id"] in doc_id_set]
    remaining_docs = [d for d in all_docs if d["doc_id"] not in doc_id_set]
    indexed, skipped, new_child_chunks = await _process_docs(target_docs, manifest)

    # Rebuild BM25: combine newly indexed chunks with chunks from remaining docs.
    # Remaining docs are already in the manifest so _process_docs only re-chunks
    # them (no embed/upsert) to collect their child chunks.
    _, _, remaining_child_chunks = await _process_docs(remaining_docs, manifest)
    all_child_chunks = new_child_chunks + remaining_child_chunks

    if all_child_chunks:
        build_index(all_child_chunks)

    elapsed_ms = (time.perf_counter() - t0) * 1_000
    _logger.info(
        "index_new_complete",
        extra={
            "indexed": len(indexed),
            "skipped": len(skipped),
            "bm25_chunks": len(all_child_chunks),
            "elapsed_ms": round(elapsed_ms, 3),
        },
    )

    return {"indexed": indexed, "skipped": skipped}
