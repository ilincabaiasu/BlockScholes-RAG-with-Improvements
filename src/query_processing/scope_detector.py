from __future__ import annotations

import re

from src.utils.logger import get_logger

_logger = get_logger(__name__)

_MONTH_MAP = {
    "january": "01", "jan": "01",
    "february": "02", "feb": "02",
    "march": "03", "mar": "03",
    "april": "04", "apr": "04",
    "may": "05",
    "june": "06", "jun": "06",
    "july": "07", "jul": "07",
    "august": "08", "aug": "08",
    "september": "09", "sep": "09",
    "october": "10", "oct": "10",
    "november": "11", "nov": "11",
    "december": "12", "dec": "12",
}

# "March 2026", "in April 2025", "during February 2024"
_MONTH_YEAR_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september"
    r"|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
    r"\s+(\d{4})\b",
    re.IGNORECASE,
)
# "Q1 2026", "Q3 2025"
_QUARTER_YEAR_RE = re.compile(r"\bq([1-4])\s+(\d{4})\b", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Patterns — ordered broadest → narrowest; first match wins
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(
    r"\b(annual|yearly|full[- ]year|year[- ]long|throughout\s+\d{4}|all of \d{4})\b",
    re.IGNORECASE,
)
_QUARTER_RE = re.compile(
    r"\b(q[1-4]|quarter(?:ly)?)\b",
    re.IGNORECASE,
)
_MONTH_PHRASE_RE = re.compile(
    r"\b(this month|last month|monthly|throughout [a-z]+|all of [a-z]+"
    r"|during [a-z]+ \d{4}|in [a-z]+ \d{4}|altogether)\b",
    re.IGNORECASE,
)
_MONTH_NAME_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september"
    r"|october|november|december"
    r"|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b",
    re.IGNORECASE,
)
# Specific day: "March 12", "12th", "on the 5th", "today", "yesterday"
_DAY_RE = re.compile(
    r"\b(\d{1,2}(?:st|nd|rd|th)|today|yesterday"
    r"|on [a-z]+ \d{1,2}|[a-z]+ \d{1,2}(?:st|nd|rd|th)?"
    r"|\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b",
    re.IGNORECASE,
)
_WEEK_RE = re.compile(
    r"\b(this week|last week|weekly|past week|recent days|these days"
    r"|past \d+ days?|over the (?:past|last) \d+ days?)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Parameter table — maps scope → retrieval knobs
# MAX_CONTEXT_TOKENS is also scaled: more temporal breadth needs more budget.
# ---------------------------------------------------------------------------

_SCOPE_PARAMS: dict[str, dict[str, int]] = {
    "day":     {"context_top_k": 5,  "reranker_top_k": 10, "max_context_tokens":  8_000},
    "week":    {"context_top_k": 10, "reranker_top_k": 15, "max_context_tokens": 10_000},
    "month":   {"context_top_k": 20, "reranker_top_k": 30, "max_context_tokens": 18_000},
    "quarter": {"context_top_k": 25, "reranker_top_k": 40, "max_context_tokens": 22_000},
    "year":    {"context_top_k": 30, "reranker_top_k": 50, "max_context_tokens": 26_000},
    "default": {"context_top_k": 10, "reranker_top_k": 20, "max_context_tokens": 12_000},
}


def detect_scope(query: str) -> str:
    """Detect the temporal scope of *query* via regex — no LLM call.

    Returns one of: "day", "week", "month", "quarter", "year", "default".

    Matching order (broadest first so a "full year" query isn't mislabelled
    as "month" just because it contains a month name):
        year → quarter → month phrase → month name → week → day → default
    """
    if _YEAR_RE.search(query):
        scope = "year"
    elif _QUARTER_RE.search(query):
        scope = "quarter"
    elif _MONTH_PHRASE_RE.search(query):
        scope = "month"
    elif _MONTH_NAME_RE.search(query):
        # Month name present — if a specific day number also appears, narrow to day
        if _DAY_RE.search(query) and re.search(r"\b\d{1,2}\b", query):
            scope = "day"
        else:
            scope = "month"
    elif _WEEK_RE.search(query):
        scope = "week"
    elif _DAY_RE.search(query):
        scope = "day"
    else:
        scope = "default"

    _logger.info("detect_scope", extra={"query": query[:80], "scope": scope})
    return scope


def scope_params(scope: str) -> dict[str, int]:
    """Return retrieval parameter overrides for *scope*.

    Keys: context_top_k, reranker_top_k, max_context_tokens.
    """
    return _SCOPE_PARAMS.get(scope, _SCOPE_PARAMS["default"])


def extract_target_period(query: str) -> str | None:
    """Extract the target date prefix (YYYY-MM) from a query, if present.

    Used to re-prioritise retrieved chunks so period-matching articles
    always appear before out-of-period articles in the context window.

    Returns e.g. "2026-03" for "March 2026", "2026-01" for "Q1 2026" first
    month, or None when no explicit period is found.
    """
    # Explicit month + year  e.g. "March 2026"
    m = _MONTH_YEAR_RE.search(query)
    if m:
        month_num = _MONTH_MAP.get(m.group(1).lower())
        if month_num:
            return f"{m.group(2)}-{month_num}"

    # Quarter + year  e.g. "Q1 2026" → first month of that quarter
    q = _QUARTER_YEAR_RE.search(query)
    if q:
        quarter_first = {"1": "01", "2": "04", "3": "07", "4": "10"}
        return f"{q.group(2)}-{quarter_first[q.group(1)]}"

    return None
