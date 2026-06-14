from __future__ import annotations

from src.chunking.fixed_chunker import chunk_fixed
from src.chunking.hierarchical_chunker import chunk_hierarchical
from src.ingestion.models import Chunk
from src.utils.token_counter import count_tokens


def chunk_document(doc: dict, mode: str) -> list[Chunk]:
    """Route a ParsedDocument dict to the appropriate chunker.

    Args:
        doc:  ParsedDocument serialised as a dict (from model_dump()).
        mode: "baseline" or "enhanced".

    Returns:
        List of Chunk objects.

    Raises:
        ValueError: if mode is not "baseline" or "enhanced".
    """
    if mode == "baseline":
        return chunk_fixed(doc)

    if mode == "enhanced":
        doc_type = doc.get("doc_type", "")

        if doc_type == "commentary" and count_tokens(doc["clean_text"]) < 600:
            return [
                Chunk(
                    chunk_id=f"{doc['doc_id']}-flat-0",
                    doc_id=doc["doc_id"],
                    parent_id=doc["doc_id"],
                    text=doc["clean_text"],
                    token_count=count_tokens(doc["clean_text"]),
                    is_parent=False,
                    chunk_index=0,
                    page_number=1,
                    has_visual_content=False,
                    pdf_path=doc.get("metadata", {}).get("source_file", ""),
                    metadata={
                        **doc.get("metadata", {}),
                        "title": doc["title"],
                        "published_date": doc["published_date"],
                        "doc_type": doc["doc_type"],
                        "asset_class": doc["asset_class"],
                    },
                )
            ]

        return chunk_hierarchical(doc)

    raise ValueError(f"mode must be 'baseline' or 'enhanced', got {mode!r}")
