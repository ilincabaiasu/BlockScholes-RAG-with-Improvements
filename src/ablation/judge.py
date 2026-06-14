"""LLM judge for the Quality Evaluation Rubric (EVAL_RUBRIC.md §3).

Scores a pipeline answer across the rubric's three retrieval dimensions and
five generation dimensions. The judge uses OpenAI (gpt-4o) regardless of the
corpus GENERATION_PROVIDER so that, when generation runs on Gemini, the judge
is a different model — avoiding the self-grading bias of a model scoring its
own output.

For queries with no time period, Temporal Accuracy is scored N/A (null) and
excluded from the retrieval denominator, per the rubric.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from src.ablation.queries import EvalQuery
from src.config.gemini_client import generate_text
from src.config.openai_client import oai_client
from src.config.settings import settings
from src.pipelines.models import QueryResult
from src.utils.logger import get_logger
from src.utils.retry import with_exponential_backoff

import openai

_logger = get_logger(__name__)

_JUDGE_MODEL = "gpt-4o"

_RUBRIC = """\
You are a strict evaluator of a financial-research RAG assistant. Score ONE answer
against the rubric below. Use the full range; reserve top bands for genuinely
excellent output. Judge only what is present — do not reward intentions.

RETRIEVAL QUALITY (based on the list of retrieved source titles+dates provided):
- retrieval_precision (0-15): proportion of sources relevant to the query's asset,
  topic and period. 13-15 nearly all relevant; 9-12 mostly; 0-8 much off-topic.
- retrieval_recall (0-15): coverage of the full query scope (full month/quarter,
  every sub-question, every asset). 13-15 full coverage; 9-12 partial; 0-8 big gaps.
- temporal_accuracy (0-10 or null): if the query has a time period, are in-period
  sources ranked before out-of-period ones and cited? 9-10 yes; 6-8 mostly; 0-5 no.
  If the query has NO explicit time period, return null.

GENERATION QUALITY (based on the answer text):
- factual_correctness (0-25): all figures traceable to cited sources; BTC/ETH not
  conflated; directions correct; absent data acknowledged not invented. 22-25 fully
  accurate; 15-21 minor non-critical errors; 0-14 material errors/hallucination.
- grounding (0-15): every factual claim has a [Source: Title | Date] inline cite;
  no outside knowledge; cited sources match the retrieved set. 13-15 all cited;
  9-12 one or two ungrounded; 0-8 frequent ungrounded claims.
- completeness (0-10): full period, all sub-questions, all assets; gaps acknowledged
  explicitly. 9-10 full; 6-8 partial; 0-5 major scope missed.
- clarity (0-5): format matches query type (factual leads with the answer; analytical
  shows reasoning; definitional explains + illustrates). 5 ideal; 3-4 minor mismatch;
  0-2 wrong format/contradictions.
- usefulness (0-5): genuine cross-source synthesis and insight, not extraction.
  5 real synthesis; 3-4 some; 0-2 pure extraction.

For out-of-corpus or uncomputable queries, the CORRECT answer is an explicit
limitation/refusal: reward factual_correctness and grounding for refusing rather
than fabricating, and score completeness on whether the limitation is acknowledged.

Return ONLY a JSON object with these exact keys (integers, or null for temporal):
retrieval_precision, retrieval_recall, temporal_accuracy, factual_correctness,
grounding, completeness, clarity, usefulness, rationale (one sentence string).
"""


@dataclass
class JudgeScores:
    retrieval_precision: int
    retrieval_recall: int
    temporal_accuracy: int | None
    factual_correctness: int
    grounding: int
    completeness: int
    clarity: int
    usefulness: int
    rationale: str

    @property
    def retrieval_total(self) -> int:
        t = self.retrieval_precision + self.retrieval_recall
        return t + (self.temporal_accuracy or 0)

    @property
    def retrieval_max(self) -> int:
        return 40 if self.temporal_accuracy is not None else 30

    @property
    def generation_total(self) -> int:
        return (
            self.factual_correctness
            + self.grounding
            + self.completeness
            + self.clarity
            + self.usefulness
        )

    @property
    def quality_total_normalised(self) -> float:
        """Quality on a 0-100 scale, renormalised when temporal is N/A."""
        raw = self.retrieval_total + self.generation_total
        max_raw = self.retrieval_max + 60
        return round(100.0 * raw / max_raw, 1)

    def to_dict(self) -> dict:
        return {
            "retrieval_precision": self.retrieval_precision,
            "retrieval_recall": self.retrieval_recall,
            "temporal_accuracy": self.temporal_accuracy,
            "factual_correctness": self.factual_correctness,
            "grounding": self.grounding,
            "completeness": self.completeness,
            "clarity": self.clarity,
            "usefulness": self.usefulness,
            "retrieval_total": self.retrieval_total,
            "retrieval_max": self.retrieval_max,
            "generation_total": self.generation_total,
            "quality_total_normalised": self.quality_total_normalised,
            "rationale": self.rationale,
        }


def _coerce_int(value, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(round(float(value)))))
    except (TypeError, ValueError):
        return lo


def _strip_json_fences(text: str) -> str:
    """Remove ```json ... ``` markdown fences some models wrap JSON in."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()


_bias_warned = False


async def _judge_call(user_msg: str) -> str:
    """Run the rubric prompt on the configured judge provider; return raw text.

    Default provider is OpenAI gpt-4o (independent of the Gemini generation
    model). When set to "gemini" the judge reuses the Gemini key — convenient,
    but if generation also runs on Gemini this is self-grading: scores skew
    high/relative, so we log a one-time warning.
    """
    global _bias_warned
    provider = settings.JUDGE_PROVIDER.lower()

    if provider == "gemini":
        if not _bias_warned and settings.GENERATION_PROVIDER.lower() == "gemini":
            _logger.warning(
                "judge_self_grading_bias",
                extra={"judge": settings.GEMINI_MODEL, "generation": settings.GEMINI_MODEL,
                       "note": "Gemini grading Gemini output — treat scores as relative, not absolute."},
            )
            _bias_warned = True
        # Gemini has no system role; the rubric is prepended inside generate_text.
        return await generate_text(_RUBRIC, user_msg, temperature=0.0)

    async def _call() -> str:
        response = await oai_client.chat.completions.create(
            model=_JUDGE_MODEL,
            messages=[
                {"role": "system", "content": _RUBRIC},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=600,
        )
        return response.choices[0].message.content

    return await with_exponential_backoff(
        _call, retry_on=openai.RateLimitError, label="judge"
    )


async def judge_answer(result: QueryResult, eq: EvalQuery) -> JudgeScores:
    """Score one pipeline run with the LLM judge. Raises on persistent failure."""
    sources_block = "\n".join(f"  - {s}" for s in (result.context_sources or [])) or "  (none)"
    has_period = eq.target_period is not None or eq.scope not in ("none",)

    user_msg = (
        f"QUERY: {eq.query}\n"
        f"QUERY TYPE: {eq.query_type}\n"
        f"EXPECTED SCOPE: {eq.scope} (target period: {eq.target_period or 'none'})\n"
        f"ANSWER IS IN CORPUS: {eq.in_corpus}\n"
        f"HAS EXPLICIT TIME PERIOD: {has_period}\n\n"
        f"RETRIEVED SOURCES (title | date):\n{sources_block}\n\n"
        f"ANSWER:\n{result.response_text or '(empty)'}\n"
    )

    raw = await _judge_call(user_msg)
    data = json.loads(_strip_json_fences(raw))

    temporal = data.get("temporal_accuracy", None)
    temporal_score = None if (temporal is None or not has_period) else _coerce_int(temporal, 0, 10)

    scores = JudgeScores(
        retrieval_precision=_coerce_int(data.get("retrieval_precision"), 0, 15),
        retrieval_recall=_coerce_int(data.get("retrieval_recall"), 0, 15),
        temporal_accuracy=temporal_score,
        factual_correctness=_coerce_int(data.get("factual_correctness"), 0, 25),
        grounding=_coerce_int(data.get("grounding"), 0, 15),
        completeness=_coerce_int(data.get("completeness"), 0, 10),
        clarity=_coerce_int(data.get("clarity"), 0, 5),
        usefulness=_coerce_int(data.get("usefulness"), 0, 5),
        rationale=str(data.get("rationale", ""))[:400],
    )
    _logger.info(
        "judge_answer",
        extra={"query_id": eq.id, "quality": scores.quality_total_normalised},
    )
    return scores