"""Streamlit UI for the Block Scholes RAG system.

Run with:
    streamlit run src/app.py
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import threading

# Ensure the project root (block-scholes-rag/) is on sys.path so that
# `import src.*` works regardless of how Streamlit was launched.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

# ---------------------------------------------------------------------------
# Secrets bridge — must run before any src.* import loads pydantic Settings.
# On Streamlit Cloud, credentials live in st.secrets (set via the UI).
# Locally they come from .env as usual; st.secrets will be empty so this
# loop is a no-op.
# ---------------------------------------------------------------------------
_SECRET_KEYS = [
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "COHERE_API_KEY",
    "QDRANT_URL",
    "QDRANT_API_KEY",
]
# Accessing st.secrets raises StreamlitSecretNotFoundError when no secrets
# file exists (e.g. local runs that use .env), so guard the whole block.
try:
    for _key in _SECRET_KEYS:
        if _key in st.secrets and not os.environ.get(_key):
            os.environ[_key] = st.secrets[_key]
except Exception:
    pass

# ---------------------------------------------------------------------------
# Persistent background event loop
# @st.cache_resource ensures this is created ONCE per server lifetime and
# survives all Streamlit reruns. All async clients (aiohttp/qdrant/gemini)
# bind to this loop on first use and stay there forever.
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_loop() -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    return loop


def _run(coro):
    """Submit *coro* to the persistent background loop and block until done."""
    return asyncio.run_coroutine_threadsafe(coro, _get_loop()).result()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Block Scholes RAG",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Block Scholes Research Assistant")
st.caption(
    "Answers grounded in Block Scholes research documents · "
    "Three pipelines compared side by side"
)

# Force long unbroken strings to wrap rather than overflow
st.markdown(
    """
    <style>
    [data-testid="stVerticalBlockBorderWrapper"] p,
    [data-testid="stVerticalBlockBorderWrapper"] div {
        word-break: break-word;
        overflow-wrap: break-word;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — corpus metadata panel
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_corpus_stats() -> dict:
    """Load corpus stats once per server lifetime (cached)."""
    from src.utils.corpus_stats import get_corpus_stats
    try:
        return _run(get_corpus_stats())
    except Exception:
        return {}


def _fmt_date(d: str) -> str:
    """YYYY-MM-DD → 'Mon YYYY'."""
    try:
        from datetime import datetime
        return datetime.strptime(d, "%Y-%m-%d").strftime("%b %Y")
    except Exception:
        return d


with st.sidebar:
    st.header("Corpus")
    _stats = _load_corpus_stats()

    if _stats:
        st.metric("Documents indexed", _stats["doc_count"])
        st.caption(
            f"**Coverage:** {_fmt_date(_stats['date_min'])} → {_fmt_date(_stats['date_max'])}"
        )
        st.divider()
        st.markdown("**Most recent articles**")
        for title, date in _stats["recent"]:
            st.markdown(f"- {title}  \n  *{_fmt_date(date)}*")
    else:
        st.caption("Corpus metadata unavailable.")

    st.divider()
    st.markdown(
        "**What this tool covers**\n"
        "- BTC & ETH derivatives research\n"
        "- Implied & realised volatility\n"
        "- ETF flows & institutional positioning\n"
        "- Options term structure & skew\n\n"
        "**What it cannot answer**\n"
        "- Live or real-time data\n"
        "- Aggregations across many documents\n"
        "- Follow-up / conversational queries\n"
        "- Events after the latest indexed date"
    )

# ---------------------------------------------------------------------------
# Citation formatting
# ---------------------------------------------------------------------------

_CITATION_RE = re.compile(r'\[Source:\s*([^\|]+?)\s*\|\s*([^\]]+?)\s*\]')

_SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _to_sup(n: int) -> str:
    return str(n).translate(_SUPERSCRIPT)


def format_citations(text: str) -> tuple[str, list[str]]:
    """Replace [Source: Title | Date] with Unicode superscript numbers.

    - Deduplicates consecutive identical citations (¹¹ → ¹)
    - Ensures a thin space before each superscript

    Returns:
        (formatted_text, numbered_source_list)
        where numbered_source_list[0] is source 1, etc.
    """
    source_index: dict[str, int] = {}
    sources_ordered: list[str] = []
    last_sup: str = ""

    def _replace(m: re.Match) -> str:
        nonlocal last_sup
        title = m.group(1).strip()
        date = m.group(2).strip()
        key = f"{title} | {date}"
        if key not in source_index:
            sources_ordered.append(f"**{title}** — {date}")
            source_index[key] = len(sources_ordered)
        sup = _to_sup(source_index[key])
        # Skip if this is the same citation as the one immediately before
        if sup == last_sup:
            return ""
        last_sup = sup
        return "\u2009" + sup  # thin space + superscript

    formatted = _CITATION_RE.sub(_replace, text)
    return formatted, sources_ordered


# ---------------------------------------------------------------------------
# Query input
# ---------------------------------------------------------------------------

query = st.text_input(
    label="Ask a question",
    placeholder="e.g. What was Bitcoin implied volatility in November 2024?",
)

run_button = st.button("Run", type="primary", disabled=not query)

# ---------------------------------------------------------------------------
# Core: run all three pipelines in a single event loop
# ---------------------------------------------------------------------------

PIPELINES = [
    ("gemini",   "🤖 Gemini Vanilla",  "No retrieval — raw model knowledge only"),
    ("baseline", "📄 Baseline RAG",    "Dense retrieval · fixed chunks · vanilla prompt"),
    ("enhanced", "🚀 Enhanced RAG",    "Hybrid retrieval · reranking · parent chunks"),
]


async def _run_all(query: str) -> dict:
    """Run all three pipelines sequentially inside one event loop."""
    from src.pipelines.gemini_pipeline import run as run_gemini
    from src.pipelines.baseline_pipeline import run as run_baseline
    from src.pipelines.enhanced_pipeline import run as run_enhanced

    runners = {
        "gemini":   run_gemini,
        "baseline": run_baseline,
        "enhanced": run_enhanced,
    }

    results = {}
    for name, runner in runners.items():
        try:
            results[name] = await runner(query)
        except Exception as exc:
            results[name] = exc
    return results


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

if run_button and query:
    with st.spinner("Running all three pipelines…"):
        all_results = _run(_run_all(query))

    cols = st.columns(3)

    for col, (pipeline_name, title, subtitle) in zip(cols, PIPELINES):
        result = all_results.get(pipeline_name)

        with col:
            st.subheader(title)
            st.caption(subtitle)

            if isinstance(result, Exception):
                err = str(result)
                if "503" in err or "UNAVAILABLE" in err or "high demand" in err.lower() or "overloaded" in err.lower():
                    st.warning("This model is currently experiencing high demand. Please try again in a moment.")
                elif "429" in err or "quota" in err.lower() or "rate limit" in err.lower():
                    st.warning("Rate limit reached. Please wait a few seconds and try again.")
                elif "401" in err or "403" in err or "API key" in err.lower():
                    st.error("Authentication error. Please check your API keys in the app settings.")
                else:
                    st.error("Something went wrong with this pipeline. Please try again.")
                continue

            # Replace inline citations with superscript numbers
            display_text, numbered_sources = format_citations(result.response_text)

            # Scrollable answer box
            # Escape bare $ signs so Streamlit doesn't treat currency
            # values (e.g. $962.8M) as LaTeX math delimiters.
            with st.container(height=500):
                st.markdown(display_text.replace("$", r"\$"))

            with st.expander("Sources & metadata"):
                if numbered_sources:
                    st.markdown("**Cited sources:**")
                    for i, src in enumerate(numbered_sources, start=1):
                        st.markdown(f"{_to_sup(i)} {src}")
                    # Also show all retrieved sources so coverage is visible
                    uncited = [s for s in result.context_sources
                               if not any(s.split(" (")[0] in ns for ns in numbered_sources)]
                    if uncited:
                        st.markdown("**Also retrieved (not cited):**")
                        for src in uncited:
                            st.markdown(f"- {src}")
                elif result.context_sources:
                    # Model didn't use inline citations but retrieval ran
                    st.markdown("**Sources retrieved (model did not cite inline):**")
                    for src in result.context_sources:
                        st.markdown(f"- {src}")
                else:
                    st.markdown("*No sources retrieved*")

                if result.visual_path_used:
                    st.markdown(
                        f"**Visual pages rendered:** {result.visual_pages_rendered}"
                    )
