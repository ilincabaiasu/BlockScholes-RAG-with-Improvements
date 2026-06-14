"""The 12 evaluation queries from EVAL_RUBRIC.md, with structured metadata.

Metadata is used by the programmatic metrics (e.g. target_period for in-period
checks, in_corpus for refusal/hallucination edge cases) and is passed to the
LLM judge as evaluation context.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalQuery:
    id: int
    query: str
    query_type: str          # factual_lookup | definitional | analytical | comparative
    category: str            # simple_fact | definitional | deep_context | comparative | ambiguous | edge_case
    scope: str               # day | week | month | quarter | none
    target_period: str | None  # "YYYY-MM" prefix expected in in-period sources, or None
    assets: list[str] = field(default_factory=list)  # ["BTC"], ["ETH"], ["BTC","ETH"], []
    in_corpus: bool = True   # False → answer is NOT in the corpus (refusal expected)
    key_criterion: str = ""


TEST_QUERIES: list[EvalQuery] = [
    EvalQuery(
        id=1,
        query="What was Bitcoin's 7-day ATM implied volatility on March 12, 2026?",
        query_type="factual_lookup",
        category="simple_fact",
        scope="day",
        target_period="2026-03",
        assets=["BTC"],
        key_criterion="Retrieval precision; factual correctness; temporal accuracy",
    ),
    EvalQuery(
        id=2,
        query="What were net ETF inflows for Bitcoin in the week of March 10, 2026?",
        query_type="factual_lookup",
        category="simple_fact",
        scope="week",
        target_period="2026-03",
        assets=["BTC"],
        key_criterion="Exact-term retrieval (BM25); grounding",
    ),
    EvalQuery(
        id=3,
        query="What is implied volatility and how is it used in crypto markets?",
        query_type="definitional",
        category="definitional",
        scope="none",
        target_period=None,
        assets=[],
        key_criterion="Response format rule; completeness; source use",
    ),
    EvalQuery(
        id=4,
        query="How did Bitcoin's implied volatility term structure evolve throughout March 2026?",
        query_type="analytical",
        category="deep_context",
        scope="month",
        target_period="2026-03",
        assets=["BTC"],
        key_criterion="Retrieval recall (month scope); temporal re-prioritisation; completeness",
    ),
    EvalQuery(
        id=5,
        query="What drove the divergence between realised and implied volatility for Ethereum in Q1 2026?",
        query_type="analytical",
        category="deep_context",
        scope="quarter",
        target_period="2026-01",
        assets=["ETH"],
        key_criterion="Quarter-scope retrieval; analytical format; synthesis across sources",
    ),
    EvalQuery(
        id=6,
        query="How did institutional positioning in Bitcoin options change after the ETF approval period?",
        query_type="analytical",
        category="deep_context",
        scope="none",
        target_period=None,
        assets=["BTC"],
        key_criterion="Multi-document synthesis; usefulness; cross-source reasoning",
    ),
    EvalQuery(
        id=7,
        query="Compare Bitcoin and Ethereum implied volatility performance in March 2026.",
        query_type="comparative",
        category="comparative",
        scope="month",
        target_period="2026-03",
        assets=["BTC", "ETH"],
        key_criterion="Both assets addressed; temporal precision; comparative format rule",
    ),
    EvalQuery(
        id=8,
        query="Compare ETH and BTC performance in Q1 2026.",
        query_type="comparative",
        category="comparative",
        scope="quarter",
        target_period="2026-01",
        assets=["BTC", "ETH"],
        key_criterion="Quarter-scope adaptive retrieval; both assets cited separately",
    ),
    EvalQuery(
        id=9,
        query="How is crypto doing lately?",
        query_type="analytical",
        category="ambiguous",
        scope="week",
        target_period=None,
        assets=[],
        key_criterion="Scope detection failure case; implicit time reference handling",
    ),
    EvalQuery(
        id=10,
        query="Is volatility high?",
        query_type="analytical",
        category="ambiguous",
        scope="none",
        target_period=None,
        assets=[],
        key_criterion="No asset, no timeframe; grounding vs hallucination test",
    ),
    EvalQuery(
        id=11,
        query="What was Bitcoin's performance in April 2027?",
        query_type="factual_lookup",
        category="edge_case",
        scope="month",
        target_period="2027-04",
        assets=["BTC"],
        in_corpus=False,
        key_criterion="Out-of-corpus query; limitation handling; no fabrication",
    ),
    EvalQuery(
        id=12,
        query="What is the average implied volatility across all March 2026 documents?",
        query_type="analytical",
        category="edge_case",
        scope="month",
        target_period="2026-03",
        assets=[],
        in_corpus=False,  # aggregation the system cannot compute → refusal expected
        key_criterion="Aggregation query; correct refusal; limitation acknowledgement",
    ),
]
