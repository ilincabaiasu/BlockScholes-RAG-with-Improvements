from __future__ import annotations

import re

from src.chunking.semantic_chunker import chunk_semantic
from src.config.settings import settings
from src.ingestion.models import Chunk
from src.utils.token_counter import count_tokens, truncate_to_tokens

_MIN_HEADING_LEN = 5
_MAX_HEADING_LEN = 80
_SHORT_DOC_TOKENS = 600
_MIN_SECTION_CHARS = 50


def _is_heading(line: str) -> bool:
    """Return True if *line* matches a section heading pattern."""
    s = line.strip()
    if not (_MIN_HEADING_LEN <= len(s) <= _MAX_HEADING_LEN):
        return False
    if s.endswith("."):
        return False
    is_all_caps = s == s.upper() and any(c.isalpha() for c in s)
    is_numbered = bool(re.match(r"^\d+\.\s+\S", s))
    is_title_case = s.istitle()
    return is_all_caps or is_numbered or is_title_case


def _split_into_sections(text: str) -> list[dict]:
    """Split *text* into sections based on heading lines.

    Returns a list of dicts: {"title": str, "text": str}.
    If no headings are found, returns one section with title="".
    """
    lines = text.splitlines()
    sections: list[dict] = []
    current_title = ""
    current_body: list[str] = []

    for line in lines:
        if _is_heading(line):
            # Save the previous section
            body = "\n".join(current_body).strip()
            if body or current_title:
                sections.append({"title": current_title, "text": body})
            current_title = line.strip()
            current_body = []
        else:
            current_body.append(line)

    # Flush the last section
    body = "\n".join(current_body).strip()
    if body or current_title:
        sections.append({"title": current_title, "text": body})

    if not sections:
        return [{"title": "", "text": text}]

    return sections


def _base_metadata(doc: dict, section_title: str) -> dict:
    return {
        **doc.get("metadata", {}),
        "title": doc["title"],
        "published_date": doc["published_date"],
        "section_title": section_title,
    }


def chunk_hierarchical(doc: dict) -> list[Chunk]:
    """Split a ParsedDocument dict into hierarchical parent/child Chunks."""
    doc_id = doc["doc_id"]
    full_text = doc["clean_text"]
    pdf_path = doc.get("metadata", {}).get("source_file", "")
    page_visual_flags: list[bool] = doc.get("metadata", {}).get("page_visual_flags", [])

    # Default visual flag for page 1 (approximation)
    visual_p1 = page_visual_flags[0] if page_visual_flags else False

    # 4. Short document — single flat chunk
    if count_tokens(full_text) < _SHORT_DOC_TOKENS:
        return [
            Chunk(
                chunk_id=f"{doc_id}-flat-0",
                doc_id=doc_id,
                parent_id=doc_id,
                text=full_text,
                token_count=count_tokens(full_text),
                is_parent=False,
                chunk_index=0,
                page_number=1,
                has_visual_content=visual_p1,
                pdf_path=pdf_path,
                metadata={
                    **doc.get("metadata", {}),
                    "title": doc["title"],
                    "published_date": doc["published_date"],
                    "doc_type": doc["doc_type"],
                    "asset_class": doc["asset_class"],
                },
            )
        ]

    # 2. Split into sections
    sections = _split_into_sections(full_text)

    chunks: list[Chunk] = []
    chunk_index = 0

    for section_index, section in enumerate(sections):
        section_text = (
            (section["title"] + "\n" + section["text"]).strip()
            if section["title"]
            else section["text"]
        )

        # 3a. Parent chunk
        parent_id = f"{doc_id}-p-{section_index}"
        parent_text = truncate_to_tokens(section_text, settings.PARENT_CHUNK_TOKENS)
        parent_chunk = Chunk(
            chunk_id=parent_id,
            doc_id=doc_id,
            parent_id=doc_id,
            text=parent_text,
            token_count=count_tokens(parent_text),
            is_parent=True,
            chunk_index=chunk_index,
            page_number=1,
            has_visual_content=visual_p1,
            pdf_path=pdf_path,
            metadata=_base_metadata(doc, section["title"]),
        )
        chunks.append(parent_chunk)
        chunk_index += 1

        # 3b. Skip child splitting for very short sections
        if len(section["text"]) < _MIN_SECTION_CHARS:
            continue

        # 3c. Semantic child chunks
        child_segments = chunk_semantic(section["text"])

        # 3d-e. Child chunks
        for child_index, segment in enumerate(child_segments):
            child_chunk = Chunk(
                chunk_id=f"{parent_id}-c-{child_index}",
                doc_id=doc_id,
                parent_id=parent_id,
                text=segment,
                token_count=count_tokens(segment),
                is_parent=False,
                chunk_index=chunk_index,
                page_number=1,
                has_visual_content=visual_p1,
                pdf_path=pdf_path,
                metadata=_base_metadata(doc, section["title"]),
            )
            chunks.append(child_chunk)
            chunk_index += 1

    return chunks
