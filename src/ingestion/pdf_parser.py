from __future__ import annotations

import os

import fitz  # pymupdf
import pdfplumber

from src.utils.logger import get_logger

_logger = get_logger(__name__)


class ParserError(Exception):
    pass


def _table_to_markdown(table: list[list]) -> str:
    """Convert a pdfplumber table (list of rows) to a pipe-delimited Markdown string."""
    if not table:
        return ""
    rows = []
    for i, row in enumerate(table):
        cells = [str(cell) if cell is not None else "" for cell in row]
        rows.append("| " + " | ".join(cells) + " |")
        if i == 0:
            rows.append("| " + " | ".join("---" for _ in cells) + " |")
    return "\n".join(rows)


def _estimate_text_area_fraction(page: pdfplumber.page.Page) -> float:
    """Return the fraction of the page area covered by word bounding boxes."""
    page_area = (page.width or 1) * (page.height or 1)
    if page_area == 0:
        return 0.0
    words = page.extract_words()
    if not words:
        return 0.0
    word_area = sum(
        (w["x1"] - w["x0"]) * (w["bottom"] - w["top"]) for w in words
    )
    return min(word_area / page_area, 1.0)


def _strip_repeated_headers_footers(pages_text: list[str]) -> list[str]:
    """Remove lines that appear on 3 or more *consecutive* pages from all pages."""
    if len(pages_text) < 3:
        return pages_text

    line_lists = [text.splitlines() for text in pages_text]

    # For each unique non-empty line, find the longest run of consecutive pages
    # that contain it. Flag lines whose longest run is >= 3.
    all_lines: set[str] = {
        line.strip()
        for lines in line_lists
        for line in lines
        if line.strip()
    }

    repeated: set[str] = set()
    for candidate in all_lines:
        max_run = current_run = 0
        for lines in line_lists:
            if candidate in {l.strip() for l in lines}:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        if max_run >= 3:
            repeated.add(candidate)

    result = []
    for lines in line_lists:
        cleaned = [l for l in lines if l.strip() not in repeated]
        result.append("\n".join(cleaned))
    return result


def parse_pdf(file_path: str) -> list[dict]:
    """Parse a PDF and return one dict per page.

    Each dict contains:
        page_number (int, 1-indexed), text (str), image_count (int),
        text_area_fraction (float), pdf_path (str).

    Raises:
        ParserError: if the file does not exist or every page is empty.
    """
    if not os.path.isfile(file_path):
        raise ParserError(f"File not found: {file_path}")

    abs_path = os.path.abspath(file_path)
    pages: list[dict] = []
    fallback_pages = 0

    fitz_doc = fitz.open(abs_path)
    try:
        with pdfplumber.open(abs_path) as pdf:
            for page_obj in pdf.pages:
                page_number = page_obj.page_number  # 1-indexed in pdfplumber

                # --- text extraction ---
                text = page_obj.extract_text() or ""

                # --- tables → markdown, appended to text ---
                tables = page_obj.extract_tables()
                for table in tables:
                    md = _table_to_markdown(table)
                    if md:
                        text = text + "\n\n" + md if text else md

                # --- image count ---
                image_count = len(page_obj.images)

                # --- text area fraction ---
                text_area_fraction = _estimate_text_area_fraction(page_obj)

                # --- pymupdf fallback for empty pages ---
                if not text.strip():
                    fallback_pages += 1
                    fitz_page = fitz_doc[page_number - 1]
                    text = fitz_page.get_text() or ""

                pages.append(
                    {
                        "page_number": page_number,
                        "text": text,
                        "image_count": image_count,
                        "text_area_fraction": text_area_fraction,
                        "pdf_path": abs_path,
                    }
                )
    finally:
        fitz_doc.close()

    if not pages:
        raise ParserError(f"No pages could be read from: {abs_path}")

    # Strip repeated headers/footers across pages
    texts = _strip_repeated_headers_footers([p["text"] for p in pages])
    for page, cleaned_text in zip(pages, texts):
        page["text"] = cleaned_text

    # Raise if every page is empty after both parsers
    if all(not p["text"].strip() for p in pages):
        raise ParserError(
            f"All pages returned empty text after pdfplumber + pymupdf: {abs_path}"
        )

    _logger.info(
        "pdf_parsed",
        extra={
            "file_path": abs_path,
            "total_pages": len(pages),
            "fallback_pages": fallback_pages,
        },
    )

    return pages
