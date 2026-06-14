from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.ingestion.cleaner import clean_page_text
from src.ingestion.deduplicator import is_duplicate, load_hashes, save_hash
from src.ingestion.metadata_extractor import extract_metadata
from src.ingestion.models import ParsedDocument
from src.ingestion.pdf_parser import parse_pdf, ParserError
from src.ingestion.visual_detector import is_visual_page
from src.utils.logger import get_logger

_logger = get_logger(__name__)

_RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
_PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


def ingest_file(file_path: str) -> dict | None:
    """Ingest a single PDF and return its ParsedDocument as a dict.

    Returns None if the file is a duplicate of an already-ingested document.
    Raises ParserError or other exceptions on hard failures — callers should
    handle these (ingest_all does so per-file).

    NOTE: the source PDF in data/raw/ is never deleted — it is required
    by the visual page renderer at query time.
    """
    # a. Parse PDF into page dicts
    pages = parse_pdf(file_path)

    # b. Clean each page and collect section titles
    cleaned_pages: list[str] = []
    all_section_titles: list[str] = []
    for page in pages:
        cleaned_text, section_titles = clean_page_text(page["text"])
        cleaned_pages.append(cleaned_text)
        all_section_titles.extend(section_titles)
        page["text"] = cleaned_text  # update in place for metadata extraction

    # c. Concatenate into full document text
    full_text = "\n\n".join(cleaned_pages)

    # d. Deduplication check
    existing_hashes = load_hashes()
    if is_duplicate(full_text, existing_hashes):
        _logger.info(
            "skipped_duplicate",
            extra={"file_path": file_path},
        )
        return None

    # e. Extract metadata
    metadata = extract_metadata(
        file_path=file_path,
        all_page_texts=cleaned_pages,
    )

    # f. Build doc_id and ParsedDocument
    doc_id = hashlib.sha256(full_text[:500].encode("utf-8")).hexdigest()[:16]

    page_visual_flags = [
        is_visual_page(page["text"], page["image_count"], page["text_area_fraction"])
        for page in pages
    ]

    doc_metadata = {
        **metadata,
        "source_file": file_path,
        "section_titles": all_section_titles,
        "page_visual_flags": page_visual_flags,
    }

    parsed_doc = ParsedDocument(
        doc_id=doc_id,
        title=metadata["title"],
        clean_text=full_text,
        published_date=metadata["published_date"],
        doc_type=metadata["doc_type"],
        asset_class=metadata["asset_class"],
        author=metadata["author"],
        metadata=doc_metadata,
    )

    # g. Persist hash
    save_hash(doc_id, full_text, existing_hashes)

    # i. Save ParsedDocument as JSON
    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _PROCESSED_DIR / f"{doc_id}.json"
    out_path.write_text(
        json.dumps(parsed_doc.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _logger.info(
        "ingested",
        extra={
            "doc_id": doc_id,
            "title": metadata["title"],
            "pages": len(pages),
            "visual_pages": sum(page_visual_flags),
            "output": str(out_path),
        },
    )

    # j. Return as dict
    return parsed_doc.model_dump()


def ingest_all() -> dict:
    """Ingest every PDF in data/raw/ and return a summary.

    Returns:
        {
            "processed": [list of doc_ids],
            "skipped":   int (duplicates),
            "failed":    [list of file paths that raised exceptions],
        }
    """
    pdf_paths = sorted(_RAW_DIR.glob("*.pdf"))

    if not pdf_paths:
        _logger.warning("ingest_all_no_pdfs", extra={"directory": str(_RAW_DIR)})
        return {"processed": [], "skipped": 0, "failed": []}

    processed: list[str] = []
    skipped = 0
    failed: list[str] = []

    for pdf_path in pdf_paths:
        file_path = str(pdf_path)
        try:
            result = ingest_file(file_path)
            if result is None:
                skipped += 1
            else:
                processed.append(result["doc_id"])
        except Exception as exc:
            _logger.error(
                "ingest_failed",
                extra={"file_path": file_path, "error": str(exc)},
            )
            failed.append(file_path)

    _logger.info(
        "ingest_all_complete",
        extra={
            "processed": len(processed),
            "skipped": skipped,
            "failed": len(failed),
        },
    )

    return {"processed": processed, "skipped": skipped, "failed": failed}
