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
    "JUDGE_PROVIDER",
    "SUPABASE_URL",
    "SUPABASE_KEY",
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


def _render_result_body(result) -> None:
    """Render the scrollable answer box + sources/metadata expander.

    Shared by the three-pipeline comparison and the customisable Enhanced block.
    """
    # Replace inline citations with superscript numbers
    display_text, numbered_sources = format_citations(result.response_text)

    # Scrollable answer box. Escape bare $ signs so Streamlit doesn't treat
    # currency values (e.g. $962.8M) as LaTeX math delimiters.
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


def _show_pipeline_error(result) -> None:
    """Map a pipeline exception to a friendly Streamlit message."""
    err = str(result)
    if "503" in err or "UNAVAILABLE" in err or "high demand" in err.lower() or "overloaded" in err.lower():
        st.warning("This model is currently experiencing high demand. Please try again in a moment.")
    elif "429" in err or "quota" in err.lower() or "rate limit" in err.lower():
        st.warning("Rate limit reached. Please wait a few seconds and try again.")
    elif "401" in err or "403" in err or "API key" in err.lower():
        st.error("Authentication error. Please check your API keys in the app settings.")
    else:
        st.error("Something went wrong with this pipeline. Please try again.")


# ---------------------------------------------------------------------------
# Query input + Enhanced-pipeline switches
# ---------------------------------------------------------------------------

# (AblationConfig flag, label, help text) in pipeline execution order. These
# switches personalise the Enhanced (3rd) pipeline in the comparison below.
ENHANCEMENT_TOGGLES = [
    ("classify",              "Query classification",       "Classify the query to choose the prompt and generation temperature."),
    ("rewrite",               "Query rewriting",            "Rewrite the query and decompose it into sub-queries."),
    ("scope_adaptive",        "Scope-adaptive breadth",     "Scale retrieval depth and token budget to the query's temporal scope."),
    ("hybrid",                "Hybrid retrieval",           "Combine dense + BM25 sparse search (RRF). Off → dense only."),
    ("rerank",                "Cross-encoder rerank",       "Re-order candidates with the Cohere reranker."),
    ("temporal_reprioritize", "Temporal re-prioritise",     "Move chunks from the query's target period to the front."),
    ("diversity_cap",         "Source diversity cap",       "Cap how many chunks come from any single document."),
    ("parent_fetch",          "Parent-context expansion",   "Expand each retrieved child chunk to its larger parent passage."),
    ("visual",                "Visual page interpretation", "Render and interpret chart/image pages with the vision model."),
]

# Map flag name → human-readable label (used for tooltip and tracker storage).
_FLAG_TO_LABEL = {flag: label for flag, label, _ in ENHANCEMENT_TOGGLES}

# Initialise in-session row accumulator on first load (persists across reruns).
if "session_rows" not in st.session_state:
    st.session_state["session_rows"] = []

query = st.text_input(
    label="Ask a question",
    placeholder="e.g. What was Bitcoin implied volatility in November 2024?",
)

st.markdown("**Enhanced pipeline (3rd column) — personalise the enhancements**")
st.caption("Everything on = the full production pipeline. Turn things off to see each component's contribution.")

flag_values: dict[str, bool] = {}
toggle_cols = st.columns(3)
for i, (flag, label, help_text) in enumerate(ENHANCEMENT_TOGGLES):
    with toggle_cols[i % 3]:
        flag_values[flag] = st.toggle(label, value=True, help=help_text, key=f"tog_{flag}")

from src.ablation.config import AblationConfig
enhanced_config = AblationConfig(name="custom", **flag_values)

run_button = st.button("Run", type="primary", disabled=not query)

# ---------------------------------------------------------------------------
# Core: run all three pipelines in a single event loop
# ---------------------------------------------------------------------------

PIPELINES = [
    ("gemini",   "🤖 Gemini Vanilla",  "No retrieval — raw model knowledge only"),
    ("baseline", "📄 Baseline RAG",    "Dense retrieval · fixed chunks · vanilla prompt"),
    ("enhanced", "🚀 Enhanced RAG",    "Personalised via the switches above"),
]


async def _run_all(query: str, enhanced_config) -> dict:
    """Run all three pipelines sequentially inside one event loop.

    The Enhanced pipeline runs with *enhanced_config* (the on/off switches);
    the other two ignore it.
    """
    from src.pipelines.gemini_pipeline import run as run_gemini
    from src.pipelines.baseline_pipeline import run as run_baseline
    from src.pipelines.enhanced_pipeline import run as run_enhanced

    results = {}
    for name, runner in (("gemini", run_gemini), ("baseline", run_baseline)):
        try:
            results[name] = await runner(query)
        except Exception as exc:
            results[name] = exc

    try:
        results["enhanced"] = await run_enhanced(query, config=enhanced_config)
    except Exception as exc:
        results["enhanced"] = exc

    return results


# ---------------------------------------------------------------------------
# Judging — every answer is scored so the Q&A feeds the performance analysis
# ---------------------------------------------------------------------------

async def _judge_results(query: str, results: dict, enhanced_result) -> dict:
    """Judge each pipeline's answer with the LLM judge. Returns {name: JudgeScores|None}."""
    from src.ablation.judge import judge_answer
    from src.ablation.queries import EvalQuery
    from src.query_processing.scope_detector import detect_scope, extract_target_period

    # Build query metadata once: reuse the enhanced pipeline's classified type
    # when available, and derive scope/period from the query text.
    qtype = getattr(enhanced_result, "query_type", None) or "analytical"
    eq = EvalQuery(
        id=0, query=query, query_type=qtype, category="live",
        scope=detect_scope(query), target_period=extract_target_period(query),
        in_corpus=True,
    )

    async def _one(name, res):
        if isinstance(res, Exception):
            return name, None
        try:
            return name, await judge_answer(res, eq)
        except Exception:
            return name, None

    pairs = await asyncio.gather(*[_one(n, r) for n, r in results.items()])
    return dict(pairs)


def _record_qa(query: str, results: dict, scores: dict, enhanced_config) -> None:
    """Append judged Q&A scores (all three pipelines) to the live tracker."""
    from datetime import datetime, timezone
    from src.ablation import tracker

    # Build a human-readable label of ENABLED feature names for the enhanced
    # pipeline, so the tooltip shows what was on rather than what was off.
    from dataclasses import fields as _dc_fields
    enabled = [
        _FLAG_TO_LABEL[f.name]
        for f in _dc_fields(enhanced_config)
        if f.name in _FLAG_TO_LABEL and getattr(enhanced_config, f.name)
    ]
    if len(enabled) == len(_FLAG_TO_LABEL):
        enh_sig = "All enhancements"
    elif enabled:
        enh_sig = " · ".join(enabled)
    else:
        enh_sig = "No enhancements"

    batch_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    rows = []
    for name, res in results.items():
        sc = scores.get(name)
        if sc is None or isinstance(res, Exception):
            continue
        latency_ms = sum(res.latency_breakdown.values()) if res.latency_breakdown else 0.0
        cv = res.citation_verification or {}
        n_hall = len(cv.get("hallucinated", [])) if isinstance(cv, dict) else 0
        rows.append({
            "batch_id": batch_id,
            "question": query,
            "pipeline": name,
            "enhanced_config": enh_sig if name == "enhanced" else "",
            "quality": sc.quality_total_normalised,
            "retrieval_pct": round(100.0 * sc.retrieval_total / sc.retrieval_max, 1) if sc.retrieval_max else None,
            "generation": sc.generation_total,
            "factual_correctness": sc.factual_correctness,
            "grounding": sc.grounding,
            "completeness": sc.completeness,
            "clarity": sc.clarity,
            "usefulness": sc.usefulness,
            "latency_ms": round(latency_ms, 1),
            "fail_flags": n_hall,
        })
    if rows:
        tracker.append_qa(rows)
        # Accumulate in-session rows so the "This session" chart updates
        # immediately without waiting for the Excel file to be re-read.
        st.session_state["session_rows"].extend(rows)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

if run_button and query:
    with st.spinner("Running all three pipelines…"):
        all_results = _run(_run_all(query, enhanced_config))

    with st.spinner("Scoring answers with the LLM judge…"):
        scores = _run(_judge_results(query, all_results, all_results.get("enhanced")))
        _record_qa(query, all_results, scores, enhanced_config)

    cols = st.columns(3)

    for col, (pipeline_name, title, subtitle) in zip(cols, PIPELINES):
        result = all_results.get(pipeline_name)

        with col:
            st.subheader(title)
            st.caption(subtitle)

            # Show which enhancements were switched off for the Enhanced column.
            if pipeline_name == "enhanced":
                disabled = enhanced_config.disabled()
                if disabled:
                    off = [lbl for fl, lbl, _ in ENHANCEMENT_TOGGLES if fl in disabled]
                    st.caption("⚠️ Disabled: " + ", ".join(off))

            if isinstance(result, Exception):
                _show_pipeline_error(result)
                continue

            sc = scores.get(pipeline_name)
            if sc is not None:
                st.metric("Judge quality", f"{sc.quality_total_normalised:.0f}/100")

            _render_result_body(result)


# ---------------------------------------------------------------------------
# Performance analysis — value quadrant + Excel-backed contribution tables
# ---------------------------------------------------------------------------

def _value_quadrant(components_df, groups_df):
    """Build the Δquality-vs-latency-cost value quadrant (Altair chart)."""
    import altair as alt
    import pandas as pd

    cols = ["label", "level", "group", "delta_quality", "latency_cost_s", "quality_per_sec", "n_pairs"]
    frames = []
    if components_df is not None and not components_df.empty:
        c = components_df.copy()
        c["label"], c["level"] = c["component"], "component"
        frames.append(c[cols])
    if groups_df is not None and not groups_df.empty:
        g = groups_df.copy()
        g["label"], g["level"], g["group"] = g["group"], "group", g["group"]
        frames.append(g[cols])
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)

    base = alt.Chart(df)
    points = base.mark_circle(opacity=0.75).encode(
        x=alt.X("latency_cost_s:Q", title="Latency cost added (s) →"),
        y=alt.Y("delta_quality:Q", title="Quality contribution (points) →"),
        size=alt.Size("level:N",
                      scale=alt.Scale(domain=["component", "group"], range=[140, 520]),
                      legend=alt.Legend(title="Level")),
        color=alt.Color("group:N", title="Group"),
        tooltip=["label", "level", "delta_quality", "latency_cost_s", "quality_per_sec", "n_pairs"],
    )
    labels = base.mark_text(align="left", dx=9, dy=-5, fontSize=11).encode(
        x="latency_cost_s:Q", y="delta_quality:Q", text="label:N",
        color=alt.Color("group:N", legend=None),
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[4, 4], color="gray").encode(y="y:Q")
    return (zero + points + labels).properties(height=460).interactive()


def _pipeline_chart(runs_df):
    """Quality-vs-latency scatter of every judged answer, coloured by pipeline."""
    import altair as alt

    df = runs_df.copy()
    df["latency_s"] = (df["latency_ms"] / 1000.0).round(2)

    # Build a human-readable "enhancements" column for the tooltip.
    # Gemini and Baseline have empty enhanced_config; give them fixed labels.
    def _enhancement_label(row):
        if row["pipeline"] == "gemini":
            return "No retrieval — raw model knowledge"
        if row["pipeline"] == "baseline":
            return "Dense retrieval · fixed chunks · vanilla prompt"
        cfg = str(row.get("enhanced_config") or "")
        return cfg if cfg else "All enhancements"
    df["enhancements"] = df.apply(_enhancement_label, axis=1)

    # Truncate long questions so the tooltip stays readable.
    df["question_short"] = df["question"].apply(
        lambda q: str(q)[:80] + ("…" if len(str(q)) > 80 else "")
    )

    points = alt.Chart(df).mark_circle(size=150, opacity=0.7).encode(
        x=alt.X("latency_s:Q", title="Latency (s) →"),
        y=alt.Y("quality:Q", title="Judge quality / 100 →", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("pipeline:N", title="Pipeline", sort=["gemini", "baseline", "enhanced"]),
        tooltip=[
            alt.Tooltip("pipeline:N", title="Pipeline"),
            alt.Tooltip("enhancements:N", title="Enhancements"),
            alt.Tooltip("question_short:N", title="Question"),
            alt.Tooltip("quality:Q", title="Quality /100"),
            alt.Tooltip("latency_s:Q", title="Latency (s)"),
            alt.Tooltip("grounding:Q", title="Grounding"),
            alt.Tooltip("retrieval_pct:Q", title="Retrieval %"),
        ],
    )
    return points.properties(height=420).interactive()


st.divider()
st.header("📈 Performance analysis")
st.caption(
    "Every question scores all three pipelines with the LLM judge. "
    "Click any dot to see which enhancements were active for that run. "
    "Upper-left is best: high quality, low latency."
)

from src.ablation import tracker as _tracker
import pandas as pd

# ---------------------------------------------------------------------------
# This session
# ---------------------------------------------------------------------------
st.subheader("This session")
_session_rows = st.session_state.get("session_rows", [])
if not _session_rows:
    st.info("Ask a question above — results appear here as soon as they're judged.")
else:
    _sess_df = pd.DataFrame(_session_rows)
    st.altair_chart(_pipeline_chart(_sess_df), use_container_width=True)
    st.caption(
        f"{len(_sess_df)} judged answers across {_sess_df['question'].nunique()} "
        f"question(s) this session."
    )

# ---------------------------------------------------------------------------
# All time (historical)
# ---------------------------------------------------------------------------
st.subheader("All time")
_qa = _tracker.load_qa()
if _qa is None:
    st.info("No historical data yet — scores accumulate in `data/ablation/qa_tracker.xlsx` as you use the app.")
else:
    _hist_runs = _qa["Runs"]
    st.altair_chart(_pipeline_chart(_hist_runs), use_container_width=True)
    st.caption(
        f"{len(_hist_runs)} judged answers across {_hist_runs['question'].nunique()} "
        f"question(s) in total."
    )
    st.markdown("**Per-pipeline averages**")
    st.dataframe(_qa["Pipelines"], use_container_width=True, hide_index=True)
    if _qa["Enhanced_Configs"] is not None and not _qa["Enhanced_Configs"].empty:
        with st.expander("Enhanced pipeline — quality by enhancement configuration"):
            st.dataframe(_qa["Enhanced_Configs"], use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Deep ablation (CLI sweep only)
# ---------------------------------------------------------------------------
_abl = _tracker.load_tracker()
if _abl is not None:
    with st.expander("🔬 Deep component & group ablation (from CLI sweep)"):
        st.caption(
            "Controlled per-component and per-group contributions vs FULL. "
            "Top-left = cheap + high impact (keep); bottom-right = slow + low impact (cut)."
        )
        _quad = _value_quadrant(_abl.get("Components"), _abl.get("Groups"))
        if _quad is not None:
            st.altair_chart(_quad, use_container_width=True)
        gcol, ccol = st.columns(2)
        with gcol:
            st.markdown("**Groups**")
            st.dataframe(_abl["Groups"], use_container_width=True, hide_index=True)
        with ccol:
            st.markdown("**Components**")
            st.dataframe(_abl["Components"], use_container_width=True, hide_index=True)
