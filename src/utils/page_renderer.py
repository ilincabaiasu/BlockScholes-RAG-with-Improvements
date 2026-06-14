from __future__ import annotations

import os
import time

from src.config.settings import settings
from src.utils.logger import get_logger

_logger = get_logger(__name__)

# Cache keyed by (pdf_path, page_number) → PIL.Image.Image
_render_cache: dict[tuple[str, int], object] = {}

# Project root (…/rag), used to resolve PDFs whose stored absolute path is stale.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RAW_DIR = os.path.join(_PROJECT_ROOT, "data", "raw")


class PageRenderError(Exception):
    pass


def _resolve_pdf_path(pdf_path: str) -> str | None:
    """Return a usable path for ``pdf_path``, or None if it cannot be found.

    The pdf_path stored in the index at index time is absolute, so it breaks if
    the project directory is moved or renamed. When the stored path is missing,
    fall back to the same filename under the current data/raw/ directory.
    """
    if os.path.isfile(pdf_path):
        return pdf_path
    candidate = os.path.join(_RAW_DIR, os.path.basename(pdf_path))
    if os.path.isfile(candidate):
        return candidate
    return None


def render_page(pdf_path: str, page_number: int) -> object:
    """Render a single PDF page and return a PIL.Image.Image.

    Results are cached in-process so repeated calls for the same
    (pdf_path, page_number) do not re-render.

    Raises:
        PageRenderError: if the PDF file is missing or the page is out of range.
    """
    cache_key = (pdf_path, page_number)
    if cache_key in _render_cache:
        return _render_cache[cache_key]

    resolved = _resolve_pdf_path(pdf_path)
    if resolved is None:
        raise PageRenderError(
            f"PDF not found: {pdf_path} — do not delete files from data/raw/"
        )
    pdf_path = resolved

    t0 = time.perf_counter()

    try:
        from pdf2image import convert_from_path

        pages = convert_from_path(
            pdf_path,
            first_page=page_number,
            last_page=page_number,
            dpi=settings.PAGE_RENDER_DPI,
        )
    except Exception as exc:
        raise PageRenderError(
            f"Failed to render page {page_number} of {pdf_path}: {exc}\n"
            "Ensure poppler is installed: conda install -c conda-forge poppler"
        ) from exc

    if not pages:
        raise PageRenderError(f"Page {page_number} out of range in {pdf_path}")

    image = pages[0]
    _render_cache[cache_key] = image

    elapsed_ms = (time.perf_counter() - t0) * 1_000
    _logger.info(
        "render_page",
        extra={
            "pdf_path": pdf_path,
            "page_number": page_number,
            "elapsed_ms": round(elapsed_ms, 3),
        },
    )

    return image


def clear_cache() -> None:
    """Clear the in-process page render cache."""
    _render_cache.clear()
