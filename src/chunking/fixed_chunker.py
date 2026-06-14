# Used exclusively by baseline_pipeline.
from __future__ import annotations

from langchain_text_splitters import TokenTextSplitter

from src.config.settings import settings
from src.ingestion.models import Chunk
from src.utils.token_counter import count_tokens

_splitter = TokenTextSplitter(
    chunk_size=settings.FIXED_CHUNK_TOKENS,
    chunk_overlap=settings.FIXED_CHUNK_OVERLAP,
    encoding_name="cl100k_base",
)


def chunk_fixed(doc: dict) -> list[Chunk]:
    """Split a ParsedDocument dict into fixed-size overlapping Chunks."""
    segments = _splitter.split_text(doc["clean_text"])
    chunks: list[Chunk] = []

    for i, segment in enumerate(segments):
        chunk = Chunk(
            chunk_id=f"{doc['doc_id']}-flat-{i}",
            doc_id=doc["doc_id"],
            parent_id=doc["doc_id"],
            text=segment,
            token_count=count_tokens(segment),
            is_parent=False,
            chunk_index=i,
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
        chunks.append(chunk)

    return chunks
