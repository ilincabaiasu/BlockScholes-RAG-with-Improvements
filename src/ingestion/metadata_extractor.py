from __future__ import annotations

import datetime
import os
import re

from src.utils.logger import get_logger

_logger = get_logger(__name__)

# Month name → zero-padded number
_MONTH_MAP = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _today_iso() -> str:
    return datetime.date.today().isoformat()


def _parse_date_string(text: str) -> str | None:
    """Try to parse a date string into YYYY-MM-DD. Return None on failure."""
    text = text.strip()

    # YYYY-MM-DD
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if m:
        try:
            datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        except ValueError:
            pass

    # PDF creation date format: D:YYYYMMDDHHmmSS or D:YYYYMMDD
    m = re.search(r"D:(\d{4})(\d{2})(\d{2})", text)
    if m:
        try:
            datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        except ValueError:
            pass

    # Month DD, YYYY  e.g. "November 4, 2024" or "Nov 4, 2024"
    m = re.search(
        r"(january|february|march|april|may|june|july|august|september|"
        r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
        r"[\s.]+(\d{1,2})[,\s]+(\d{4})",
        text,
        re.IGNORECASE,
    )
    if m:
        month = _MONTH_MAP.get(m.group(1).lower())
        if month:
            try:
                datetime.date(int(m.group(3)), int(month), int(m.group(2)))
                return f"{m.group(3)}-{month}-{int(m.group(2)):02d}"
            except ValueError:
                pass

    # DD Month YYYY  e.g. "4 November 2024"
    m = re.search(
        r"(\d{1,2})[\s.]+(january|february|march|april|may|june|july|august|"
        r"september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|"
        r"oct|nov|dec)[\s.,]+(\d{4})",
        text,
        re.IGNORECASE,
    )
    if m:
        month = _MONTH_MAP.get(m.group(2).lower())
        if month:
            try:
                datetime.date(int(m.group(3)), int(month), int(m.group(1)))
                return f"{m.group(3)}-{month}-{int(m.group(1)):02d}"
            except ValueError:
                pass

    return None


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def extract_metadata(
    file_path: str,
    all_page_texts: list[str],
    pdf_title_from_properties: str = "",
    pdf_author_from_properties: str = "",
    pdf_creation_date: str = "",
) -> dict:
    """Extract document metadata from file path, PDF properties, and page text.

    Returns a dict with keys: title, published_date, doc_type, asset_class, author.
    """
    stem = os.path.splitext(os.path.basename(file_path))[0]
    page1_text = all_page_texts[0] if all_page_texts else ""

    # ------------------------------------------------------------------
    # title
    # ------------------------------------------------------------------
    if pdf_title_from_properties and len(pdf_title_from_properties.strip()) > 5:
        title = pdf_title_from_properties.strip()
    else:
        title = re.sub(r"[_\-]+", " ", stem).strip().title()

    # ------------------------------------------------------------------
    # published_date
    # ------------------------------------------------------------------
    published_date: str | None = None

    # 1. YYYY-MM-DD in filename stem
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", stem)
    if m:
        published_date = _parse_date_string(m.group(0))

    # 2. PDF creation date property
    if not published_date and pdf_creation_date:
        published_date = _parse_date_string(pdf_creation_date)

    # 3. Date patterns in first 800 chars of page 1
    #    Block Scholes docs place the date below the author line, which
    #    can appear ~370-400 chars in after title + abstract + author names.
    if not published_date and page1_text:
        published_date = _parse_date_string(page1_text[:800])

    # 4. Fall back to today
    if not published_date:
        _logger.warning(
            "metadata_date_fallback",
            extra={"file_path": file_path, "fallback": "today"},
        )
        published_date = _today_iso()

    # ------------------------------------------------------------------
    # doc_type
    # ------------------------------------------------------------------
    scan_doc_type = (stem + " " + page1_text[:200]).lower()
    total_words = sum(len(t.split()) for t in all_page_texts)

    if any(kw in scan_doc_type for kw in ("methodology", "model", "technical")):
        doc_type = "model-doc"
    elif total_words < 800:
        doc_type = "commentary"
    elif any(kw in scan_doc_type for kw in ("report",)):
        doc_type = "report"
    else:
        doc_type = "article"

    # ------------------------------------------------------------------
    # asset_class
    # ------------------------------------------------------------------
    scan_asset = (title + " " + page1_text[:500]).lower()

    asset_class = "unknown"
    if re.search(r"\bbtc\b|bitcoin", scan_asset):
        asset_class = "BTC"
    elif re.search(r"\beth\b|ethereum", scan_asset):
        asset_class = "ETH"
    elif re.search(r"\boptions\b|\bvolatility\b|\bvol\b", scan_asset):
        asset_class = "options"
    elif re.search(r"cross-asset|multi-asset", scan_asset):
        asset_class = "cross-asset"

    # ------------------------------------------------------------------
    # author
    # ------------------------------------------------------------------
    author = (
        pdf_author_from_properties.strip()
        if pdf_author_from_properties.strip()
        else "Block Scholes"
    )

    return {
        "title": title,
        "published_date": published_date,
        "doc_type": doc_type,
        "asset_class": asset_class,
        "author": author,
    }
