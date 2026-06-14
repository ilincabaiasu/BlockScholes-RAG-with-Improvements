from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ParsedDocument(BaseModel):
    model_config = ConfigDict(extra="ignore")

    doc_id: str          # sha256 of clean_text[:500], hex string
    title: str
    clean_text: str
    published_date: str  # ISO-8601 date string e.g. "2024-11-04"
    doc_type: str        # "report" | "commentary" | "model-doc" | "article"
    asset_class: str     # "BTC" | "ETH" | "cross-asset" | "options" | "unknown"
    author: str = "Block Scholes"
    metadata: dict = {}


class Chunk(BaseModel):
    model_config = ConfigDict(extra="ignore")

    chunk_id: str        # f"{doc_id}-{chunk_index}"
    doc_id: str
    parent_id: str       # doc_id for flat chunks; section chunk_id for children
    text: str
    token_count: int
    is_parent: bool = False
    section_title: str = ""
    chunk_index: int
    page_number: int = 1
    has_visual_content: bool = False
    pdf_path: str = ""   # absolute path to source PDF in data/raw/
    metadata: dict = {}  # inherits all ParsedDocument fields


class ContextChunk(BaseModel):
    model_config = ConfigDict(extra="ignore")

    parent_text: str
    child_text: str
    metadata: dict
    reranker_score: float
    visual_path_used: bool = False
    source_title: str = ""   # article title for citation
    source_date: str = ""    # published_date for citation
