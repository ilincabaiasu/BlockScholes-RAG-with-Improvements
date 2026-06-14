"""Programmatic (no-LLM) metrics computed from a QueryResult.

These are deterministic, cheap signals derived from the pipeline output and the
expected query metadata. They cover the parts of EVAL_RUBRIC.md that can be
checked mechanically: citation format/grounding, source diversity, in-period
retrieval, temporal prioritisation, refusal handling, and latency. Subjective
rubric dimensions are left to the LLM judge.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from src.ablation.queries import EvalQuery
from src.pipelines.models import QueryResult

# Inline citation in the answer body, e.g. "[Source: BTC Vol Report | 2026-03-12]"
_INLINE_CITE_RE = re.compile(r"\[Source:\s*([^\|\]]+?)\s*\|\s*([^\]]+?)\s*\]")
# A date prefix anywhere in a source string, e.g. "Title (2026-03-12)" → 2026-03
_DATE_RE = re.compile(r"(\d{4}-\d{2})(?:-\d{2})?")
# Specific numeric claims: percentages, prices, multi-digit figures, $-amounts.
_NUMERIC_RE = re.compile(r"(?<![\w-])(?:\$?\d[\d,]*\.?\d*\s?%?)(?![\w-])")
# Phrases signalling an explicit limitation / refusal.
_REFUSAL_RE = re.compile(
    r"(do(?:es)?\s+not\s+contain|no\s+(?:data|information|documents?|sources?)"
    r"|not\s+(?:available|present|found|in\s+the\s+corpus)|cannot\s+(?:be\s+)?"
    r"(?:compute|calculate|provide|determine)|unable\s+to|outside\s+the\s+"
    r"(?:corpus|scope|available)|no\s+in-?(?:scope|period)|i\s+don'?t\s+have)",
    re.IGNORECASE,
)
# "does not contain ..." immediately followed by substantive content → contradiction.
_OMISSION_CONTRADICTION_RE = re.compile(
    r"(do(?:es)?\s+not\s+contain|corpus\s+does\s+not\s+include)[^.]{0,80}\.\s+\S",
    re.IGNORECASE,
)


@dataclass
class Metrics:
    # Retrieval / sources
    n_sources: int
    n_distinct_docs: int
    in_period_present: bool       # at least one source within target_period
    in_period_first: bool         # the first-ranked source is in-period
    # Citation / grounding
    n_inline_citations: int
    n_hallucinated_citations: int
    citation_format_consistent: bool   # ≥1 inline cite and none hallucinated
    numeric_claims_uncited: bool       # has figures but zero inline citations
    # Format / content
    n_sentences: int
    both_assets_covered: bool          # comparative: BTC and ETH both mentioned
    refused: bool                      # explicit limitation acknowledged
    omission_contradiction: bool       # "does not contain" then provides it anyway
    # Performance
    total_latency_ms: float
    # Derived
    auto_fail_flags: list

    def to_dict(self) -> dict:
        return asdict(self)


def _source_periods(sources: list[str]) -> list[str]:
    """Extract YYYY-MM prefixes from 'Title (date)' source strings."""
    periods = []
    for s in sources:
        m = _DATE_RE.search(s)
        if m:
            periods.append(m.group(1))
    return periods


def compute_metrics(result: QueryResult, eq: EvalQuery) -> Metrics:
    """Compute deterministic metrics for one pipeline run."""
    text = result.response_text or ""
    sources = result.context_sources or []

    # --- sources / retrieval ---
    n_sources = len(sources)
    n_distinct_docs = len({s.split(" (")[0].strip().lower() for s in sources})
    periods = _source_periods(sources)
    tp = eq.target_period
    in_period_present = bool(tp) and any(p == tp for p in periods)
    in_period_first = bool(tp) and bool(periods) and periods[0] == tp

    # --- citations / grounding ---
    inline = _INLINE_CITE_RE.findall(text)
    n_inline = len(inline)
    cv = result.citation_verification or {}
    hallucinated = cv.get("hallucinated", []) if isinstance(cv, dict) else []
    n_hallucinated = len(hallucinated) if isinstance(hallucinated, list) else 0
    citation_format_consistent = n_inline > 0 and n_hallucinated == 0

    has_numbers = bool(_NUMERIC_RE.search(text))
    numeric_claims_uncited = has_numbers and n_inline == 0

    # --- format / content ---
    n_sentences = len([s for s in re.split(r"[.!?]+", text) if s.strip()])
    low = text.lower()
    both_assets_covered = (
        ("bitcoin" in low or "btc" in low) and ("ethereum" in low or "eth" in low)
    )
    refused = bool(_REFUSAL_RE.search(text))
    omission_contradiction = bool(_OMISSION_CONTRADICTION_RE.search(text))

    # --- performance ---
    total_latency_ms = float(sum(result.latency_breakdown.values())) if result.latency_breakdown else 0.0

    # --- auto-fail flags (EVAL_RUBRIC.md §2.5) ---
    flags: list[str] = []
    if omission_contradiction:
        flags.append("omission_contradiction")
    if n_hallucinated > 0:
        flags.append("hallucinated_citation")
    if numeric_claims_uncited:
        flags.append("uncited_numeric_claims")
    if eq.category in ("deep_context", "comparative") and n_sentences < 3:
        flags.append("too_short_for_analytical")
    if eq.target_period and periods and not in_period_present and eq.in_corpus:
        flags.append("only_out_of_period_sources")
    if not eq.in_corpus and not refused:
        flags.append("no_refusal_on_out_of_corpus")

    return Metrics(
        n_sources=n_sources,
        n_distinct_docs=n_distinct_docs,
        in_period_present=in_period_present,
        in_period_first=in_period_first,
        n_inline_citations=n_inline,
        n_hallucinated_citations=n_hallucinated,
        citation_format_consistent=citation_format_consistent,
        numeric_claims_uncited=numeric_claims_uncited,
        n_sentences=n_sentences,
        both_assets_covered=both_assets_covered,
        refused=refused,
        omission_contradiction=omission_contradiction,
        total_latency_ms=round(total_latency_ms, 1),
        auto_fail_flags=flags,
    )
