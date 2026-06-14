# Evaluation Framework
## Block Scholes RAG System — Three-Pipeline Comparison
### IBRAU9 Individual Assignment 2

---

## 1. Introduction

This evaluation framework is designed for assessing outputs from the three pipelines of the Block Scholes Research Assistant across a standardised set of test queries:

- **Gemini Vanilla** — raw model knowledge, no retrieval
- **Baseline RAG** — dense-only retrieval, fixed 500-token chunks, minimal prompt
- **Enhanced RAG** — hybrid retrieval (dense + BM25), RRF fusion, Cohere cross-encoder reranking, parent-child chunking, adaptive scope detection, temporal re-prioritisation, and query-type-structured prompting

The framework is structured in two layers. First, a Task Compliance Checklist determines whether each pipeline's output satisfies the minimum output requirements for the given query. Second, a Quality Evaluation Rubric assesses the output across both retrieval dimensions (precision, recall, temporal accuracy) and generation dimensions (correctness, grounding, completeness, clarity, usefulness). A final workflow explains how the two layers combine into a per-pipeline verdict and a cross-pipeline comparison.

The framework is applied independently to each of the three pipelines on the same query, allowing direct side-by-side comparison.

---

## 2. Task Compliance Checklist

### 2.1 Purpose

The Task Compliance Checklist determines whether each pipeline's output satisfies the minimum structural and content requirements for the given query. These are necessary conditions — a pass does not guarantee quality.

### 2.2 Scoring Rules

| Rule | Description |
|---|---|
| Binary scoring | Each item is scored as Met = full points or Not met = 0 |
| No partial credit | No partial credit is allowed |
| Pipeline-conditional items | Items marked *(RAG only)* are scored N/A for Gemini Vanilla and do not count against its total |
| Query-type-conditional items | Items marked with a query type apply only when that query type is being evaluated; score N/A otherwise |
| Contradiction items | For items phrased as "does not...", score Met only if the prohibited behaviour is entirely absent |
| Duplicated items | Preserve all items exactly as written; do not merge similar requirements |
| Total possible score | 52 points for Baseline and Enhanced RAG; 36 points for Gemini Vanilla |

### 2.3 Summary of Compliance Categories

| Section | Category | Points (RAG) | Points (Vanilla) |
|---|---|---|---|
| A | Retrieval Quality | 16 | N/A |
| B | Citation and Grounding | 10 | 10 |
| C | Factual Accuracy | 12 | 12 |
| D | Response Format | 8 | 8 |
| E | Completeness | 6 | 6 |
| **Total** | | **52** | **36** |

---

### A. Retrieval Quality — 16 points *(RAG only)*

#### A1. Relevance

| Points | Requirement |
|---|---|
| +2 | At least one retrieved source is directly relevant to the topic and domain of the query |
| +2 | For time-scoped queries (day / week / month / quarter): at least one in-period source — published within the stated period — is present in the retrieved set |
| +2 | Retrieved sources span at least two distinct documents; context window is not dominated by chunks from a single article |
| +2 | No retrieved source is entirely off-topic or from a clearly unrelated domain or asset class |

#### A2. Temporal and Scope Accuracy

| Points | Requirement |
|---|---|
| +2 | For month or quarter-scoped queries: the retrieved set includes sources from at least two separate weeks or months within the stated period |
| +2 | For time-scoped queries: in-period sources appear before out-of-period sources in the context window (temporal prioritisation is correctly applied) |

#### A3. Retrieval Prohibited Errors

| Points | Requirement |
|---|---|
| +2 | Does not present out-of-period sources as primary results when in-period sources exist in the corpus |
| +2 | Does not return zero retrieved sources for a query whose answer exists within the indexed corpus |

---

### B. Citation and Grounding — 10 points

#### B1. Citation Presence and Format

| Points | Requirement |
|---|---|
| +2 | Every specific numerical claim (price level, implied volatility figure, ETF flow volume, percentage change) carries an inline citation |
| +2 | Every market event or development stated as fact carries an inline citation |
| +2 | Citations consistently follow the format \[Source: Title \| Date\] throughout the response |

#### B2. Grounding Integrity

| Points | Requirement |
|---|---|
| +2 | Response does not introduce factual claims from outside the retrieved documents or from the model's parametric training memory |
| +2 | Response does not fabricate a source title or publication date not present in the retrieval pool |

---

### C. Factual Accuracy — 12 points

#### C1. Numerical and Attribution Accuracy

| Points | Requirement |
|---|---|
| +2 | Specific numerical values cited are consistent with the stated source document |
| +2 | Bitcoin and Ethereum claims are correctly attributed and not conflated |
| +2 | Dates and time periods stated in the response are accurate relative to the cited sources |

#### C2. Market Mechanics and Consistency

| Points | Requirement |
|---|---|
| +2 | Descriptions of implied volatility, ETF flows, term structure, or options mechanics are directionally correct |
| +2 | Does not assert that data exists for a period when no in-scope documents were retrieved |
| +2 | Does not contradict a cited source within the same response |

---

### D. Response Format — 8 points

#### D1. Format Matches Query Type

*Score the two items that apply to the query being evaluated.*

| Points | Query Type | Requirement |
|---|---|---|
| +2 | Factual lookup | Response leads directly with the answer and the supporting figure; is concise; does not pad with unnecessary background before delivering the fact |
| +2 | Definitional | Response defines the concept, explains the mechanics, and illustrates with at least one cited data point from the corpus — does not give a one-sentence definition when richer context exists |
| +2 | Analytical | Response uses a visible reasoning structure — for example: Observation, Evidence, Interpretation — rather than presenting disconnected assertions |
| +2 | Comparative | Each item in the comparison is addressed in a dedicated section with individual inline citations; if one item lacks data, this is explicitly stated and the other item is still fully addressed |

#### D2. Format Prohibited Errors

| Points | Requirement |
|---|---|
| +2 | Does not produce a one-sentence response to an analytical or comparative query |
| +2 | Does not produce a generic or template-like response that could apply to any query regardless of the specific question asked |

---

### E. Completeness — 6 points

#### E1. Scope Coverage

| Points | Requirement |
|---|---|
| +2 | For time-scoped queries: response covers the full stated period, not just a single date or week within it |
| +2 | For multi-part queries: every distinct sub-question is addressed with sourced content |

#### E2. Errors of Omission

| Points | Requirement |
|---|---|
| +2 | Does not open with "the documents do not contain this information" and then proceed to provide that information — a logical contradiction that undermines credibility |

---

### 2.4 Compliance Score Interpretation

| Score Range (RAG) | Score Range (Vanilla) | Interpretation |
|---|---|---|
| 48–52 | 33–36 | Strong compliance — output meets all minimum requirements |
| 38–47 | 26–32 | Adequate — minor gaps; core requirements met |
| 26–37 | 18–25 | Material compliance failures — systematic issues in at least one section |
| Below 26 | Below 18 | Weak compliance — output fails multiple minimum requirements |

---

### 2.5 Recommended Automatic Fail Flags

Even if the numeric compliance score is not low, flag the pipeline output for mandatory review if any of the following apply:

- Response contains "does not contain" or "the corpus does not include" immediately followed by relevant factual content from that same corpus
- Response cites a source title or date not present in the retrieval pool
- Response provides specific numerical data (figures, percentages, volumes) with no inline citation anywhere in the response
- Response is shorter than three sentences for an analytical or comparative query
- For time-scoped queries: response discusses only events outside the stated period
- Response fails to address the query topic at all and instead returns a refusal or a definition of a different concept

---

## 3. Quality Evaluation Rubric

### 3.1 Purpose

The Quality Evaluation Rubric is applied after the Task Compliance Checklist. Its purpose is to assess whether the output is not only compliant, but accurate, grounded, complete, and genuinely useful. A response can pass every binary checklist item and still score poorly on quality.

The rubric is divided into two domains that mirror the two-stage architecture of the RAG system:

- **Retrieval Quality (40 points):** evaluates whether the pipeline surfaced the right documents — measuring precision, recall, and temporal accuracy independently.
- **Generation Quality (60 points):** evaluates what the pipeline did with those documents — measuring factual correctness, grounding, completeness, clarity, and usefulness.

**Total quality score: 100 points.**

*For Gemini Vanilla, Retrieval Quality is scored 0/40 by definition (no retrieval occurs). Only Generation Quality applies.*

---

### 3.2 Retrieval Quality Rubric — 40 points

| Criterion | Weight | What It Means in This Task | Why It Matters | Strong Performance | Weak Performance |
|---|---|---|---|---|---|
| **Retrieval Precision** | 15 | Of the chunks that appear in the context window, the proportion that are actually relevant to the query. Precision penalises noise — chunks that are topically adjacent but do not contribute to answering what was asked, or that belong to the wrong asset or the wrong time period. | Low precision dilutes the context window. When irrelevant chunks displace relevant ones within the token budget, the generation model has less evidence to work with and is more likely to produce vague or hallucinated answers. | Context window contains mostly relevant chunks; irrelevant or tangentially related content is minimal; every chunk in the window plausibly contributes to answering the query | Several retrieved chunks are about a different asset, a different time period, or a different concept from what was asked; context window contains filler that does not contribute to the answer |
| **Retrieval Recall** | 15 | Whether the retrieved set covers the full scope of what the query requires. For a month-scoped query, recall measures whether the pipeline retrieved sources spanning the full month rather than clustering on a single event or a single week. For a multi-part query, recall measures whether the retrieval covered each sub-question. | Low recall produces incomplete answers regardless of how well the generation model performs. If the retrieval stage misses half the relevant documents, the LLM cannot synthesise what it was never given. | For month/quarter queries: retrieved sources span the full period with meaningful coverage across different weeks; for multi-part queries: relevant documents for each sub-question are present in the retrieved pool; no large coverage gap for any part of the query | Retrieved documents cluster on a single event or date within a broader query period; multiple sub-questions have no supporting sources in the context window; significant portions of the query scope are unrepresented in the retrieval |
| **Temporal Accuracy** | 10 | For queries that specify a time period: whether in-period sources are ranked above out-of-period sources in the context window. This criterion specifically tests the temporal re-prioritisation mechanism — whether documents published in the queried month or quarter appear before semantically similar but out-of-period documents. | Cross-encoders score semantic relevance, not temporal relevance. Without temporal re-prioritisation, a semantically similar article from the wrong month can displace a less-fluently worded but temporally correct article. The result is a factually misleading answer that cites sources outside the queried period. | All or most in-period sources appear before out-of-period sources in the context window; the answer cites sources from the correct period; out-of-period sources are present only as supplementary context after in-period sources are exhausted | Out-of-period sources appear ahead of in-period sources; the generated answer cites articles from the wrong month or quarter; in-period sources were retrieved but pushed outside the context window by higher-scoring out-of-period content |

---

### 3.3 Generation Quality Rubric — 60 points

| Criterion | Weight | What It Means in This Task | Why It Matters | Strong Performance | Weak Performance |
|---|---|---|---|---|---|
| **Factual Correctness** | 25 | All specific claims — implied volatility figures, ETF flow volumes, spot prices, dates, market directions — are accurate relative to the cited source documents. No fabricated data. Figures presented in the response can be traced directly to a retrieved document. Where exact data is absent, the model says so explicitly rather than estimating or generalising. | In financial research, an inaccurate answer is worse than no answer. A wrong implied volatility figure or a misattributed ETF flow can directly mislead analytical conclusions. Correctness is the minimum standard for a research assistant to be trusted. | All specific values match their cited sources; BTC and ETH are never conflated; market directions and dynamics are described accurately; where data is absent for part of the query, the model explicitly states this rather than filling the gap with invention | Hallucinated numerical figures; wrong asset attributed to the wrong date or source; market direction stated incorrectly (e.g., rising described as falling); specific data presented as fact with no traceable origin in any retrieved document |
| **Grounding and Source Use** | 15 | Every factual claim is anchored to a specific retrieved document using the \[Source: Title \| Date\] inline citation format. The model draws exclusively from retrieved context and does not introduce knowledge from its parametric training memory. The number of cited sources reflects the breadth of evidence actually available. | RAG exists to prevent hallucination and to ground answers in verifiable sources. An uncited claim is unverifiable regardless of whether it happens to be correct. Grounding is what distinguishes a RAG system from a vanilla language model on factual financial queries. | All factual claims carry inline citations; the citation format is consistent throughout the response; sources cited in the text match the retrieved set; no claims are introduced from outside the corpus; the model distinguishes clearly between retrieved evidence and general context | Claims appear without citation; the model introduces terminology, figures, or events not present in the retrieved documents; citation format is inconsistently applied or absent for the most important claims; cited sources cannot be matched to any document in the retrieval pool |
| **Completeness** | 10 | The answer addresses the full scope of what was asked — all sub-questions, the full stated time period, and all assets mentioned in a comparative query. Partial answers that address only one part of a multi-part query are penalised proportionally to the size of the gap. Gaps that are explicitly acknowledged score better than gaps that are silently omitted. | Incomplete answers mislead by omission. A response covering only Bitcoin in a Bitcoin-Ethereum comparison, or only one week of data for a full-month query, fails the user even if what it does cover is accurate. The reader has no way of knowing what is missing unless the model tells them. | Full stated period is covered for time-scoped queries; all assets addressed in comparative queries; every sub-question answered with sourced evidence; when the corpus does not contain data for part of the query, the model explicitly acknowledges this rather than ignoring it | Only part of the stated time period is discussed with no acknowledgement that coverage is partial; one asset is missing from a comparative query; multiple sub-questions are not addressed; the model silently omits significant parts of the query scope |
| **Clarity and Format** | 5 | The response is clearly written and structured in a way that matches the query type. Factual queries lead directly with the answer. Analytical queries show visible reasoning structure. Definitional queries explain the concept, its mechanics, and illustrate with concrete cited data. Terminology is used consistently. | Format signals whether the model understood what was being asked. A one-sentence response to a complex analytical question about a month of volatility dynamics indicates the model did not engage with the query at the appropriate depth. | Format matches the query type; analytical responses show structured reasoning; definitional responses illustrate concepts with data rather than offering a single sentence; terminology is consistent throughout; the reader can follow the logical chain from claim to citation | One-sentence definitional answers when full context was available; unstructured analytical responses; contradictions in terminology across the same response; the reader cannot determine which claim follows from which source |
| **Practical Usefulness** | 5 | The answer synthesises across multiple retrieved sources to produce an insight that the reader could not have obtained by reading a single document. It identifies patterns, trends, or comparisons. It contextualises specific data points within the broader market picture rather than restating retrieved text verbatim. | A research assistant exists to produce analytical value, not to return a formatted citation list. A response that concatenates retrieved sentences without synthesis provides no value beyond what a keyword search would return. | Synthesises data points across sources to identify a trend, a contrast, or a cause; contextualises figures within the broader market picture; the reader gains an understanding they could not get from reading the raw documents alone | Pure text extraction with no synthesis; response reads as a sequence of retrieved sentences with citations; no pattern or trend identified; reader would need to read the source documents themselves to understand what the response means |

---

### 3.4 Quality Scoring Guides by Criterion

#### Retrieval Precision — 15 points
- **13–15:** Context window is predominantly relevant; all or nearly all chunks contribute to answering the query
- **9–12:** Mostly relevant; a few irrelevant chunks present but do not dominate
- **0–8:** Significant proportion of the context window is off-topic or from the wrong period

#### Retrieval Recall — 15 points
- **13–15:** Retrieved set covers the full query scope; multi-part queries have supporting sources for every sub-question
- **9–12:** Partial coverage; one sub-question or part of the time period is underrepresented
- **0–8:** Significant coverage gaps; major portions of the query scope have no supporting sources

#### Temporal Accuracy — 10 points
- **9–10:** In-period sources appear first; answer cites correct-period documents; temporal re-prioritisation is visibly effective
- **6–8:** Mostly correct; one or two out-of-period sources appear ahead of in-period ones but do not dominate
- **0–5:** Out-of-period sources dominate the context window; answer cites the wrong period

*Score 0/10 for general queries with no time period (N/A — criterion does not apply; exclude from denominator).*

#### Factual Correctness — 25 points
- **22–25:** Fully accurate; all figures traceable to cited sources; no fabrication; absent data explicitly noted
- **15–21:** Mostly accurate; minor non-critical errors that do not change the substance of the answer
- **0–14:** Material factual errors; hallucinated figures; claims that directly contradict their cited source

#### Grounding and Source Use — 15 points
- **13–15:** Every factual claim is cited; format is consistent; no outside knowledge introduced
- **9–12:** Most claims cited; one or two ungrounded statements; format mostly consistent
- **0–8:** Frequent ungrounded claims; citation-free responses; outside knowledge material to the answer

#### Completeness — 10 points
- **9–10:** Full query scope addressed; all sub-questions answered; gaps explicitly acknowledged
- **6–8:** Partial coverage; some sub-questions or part of the period unaddressed
- **0–5:** Significant scope missed; silent truncation; major sub-questions absent

#### Clarity and Format — 5 points
- **5:** Format appropriate to query type; reasoning easy to follow; terminology consistent
- **3–4:** Readable but minor format mismatch or uneven depth
- **0–2:** Format does not match query type; contradictions; reader cannot trace the reasoning

#### Practical Usefulness — 5 points
- **5:** Genuine synthesis; identifies patterns or trends; reader gains analytical insight
- **3–4:** Some synthesis; mostly extraction with limited cross-source reasoning
- **0–2:** No synthesis; pure extraction; no analytical value added

---

### 3.5 Quality Score Interpretation

| Score Range | Interpretation |
|---|---|
| 88–100 | High-quality research assistant output — accurate, grounded, complete, and analytically useful |
| 72–87 | Useful output with meaningful gaps in at least one criterion |
| 55–71 | Weak output quality — systematic issues in retrieval or generation |
| Below 55 | Not fit for research use |

---

## 4. Evaluation Workflow

### 4.1 Step 1: Run the Task Compliance Checklist

**Purpose:** Determine whether each pipeline's output satisfies the minimum structural and content requirements for the given query.

**Method:**
- Score every applicable binary item for each pipeline
- Sum points out of 52 (RAG) or 36 (Vanilla)
- Record any automatic fail flags
- Record the top three missing items by importance

**Output of Step 1:**
- Compliance score: __ / 52
- Automatic fail flags: Yes / No
- Top missing items: 1. &nbsp;&nbsp; 2. &nbsp;&nbsp; 3.
- Checklist verdict: Strong / Adequate / Weak

---

### 4.2 Step 2: Run the Quality Evaluation Rubric

**Purpose:** Assess whether the output is not just compliant, but accurate, grounded, complete, and analytically useful across both retrieval and generation dimensions.

**Method:**
- Score each retrieval criterion (Precision, Recall, Temporal Accuracy)
- Score each generation criterion (Correctness, Grounding, Completeness, Clarity, Usefulness)
- Sum to 100
- Add a one-sentence comment per criterion explaining the primary reason for the score

**Output of Step 2:**
- Retrieval quality: __ / 40
- Generation quality: __ / 60
- Quality total: __ / 100
- Criterion breakdown: listed below
- Main quality concerns: 3–5 bullet points maximum

---

### 4.3 Step 3: Form the Cross-Pipeline Comparison

Complete the following table for each query evaluated:

| | Gemini Vanilla | Baseline RAG | Enhanced RAG |
|---|---|---|---|
| **Compliance** | __ / 36 | __ / 52 | __ / 52 |
| **Fail Flags** | Yes / No | Yes / No | Yes / No |
| **Retrieval Precision** | N/A | __ / 15 | __ / 15 |
| **Retrieval Recall** | N/A | __ / 15 | __ / 15 |
| **Temporal Accuracy** | N/A | __ / 10 | __ / 10 |
| **Retrieval Total** | 0 / 40 | __ / 40 | __ / 40 |
| **Factual Correctness** | __ / 25 | __ / 25 | __ / 25 |
| **Grounding** | __ / 15 | __ / 15 | __ / 15 |
| **Completeness** | __ / 10 | __ / 10 | __ / 10 |
| **Clarity & Format** | __ / 5 | __ / 5 | __ / 5 |
| **Usefulness** | __ / 5 | __ / 5 | __ / 5 |
| **Generation Total** | __ / 60 | __ / 60 | __ / 60 |
| **Quality Total** | __ / 60 | __ / 100 | __ / 100 |
| **Key Strength** | | | |
| **Key Weakness** | | | |

---

### 4.4 Step 4: Form the Final Verdict per Pipeline

| Verdict | Conditions |
|---|---|
| **Strong** | No fail flags; Compliance ≥ 46/52; Factual Correctness ≥ 20/25; Quality total ≥ 82/100 |
| **Adequate** | No fail flags; Compliance 36–45/52 and/or Quality 65–81/100; core requirements met with gaps |
| **Weak** | Any fail flag; Compliance < 36/52; Factual Correctness < 15/25; or response is largely ungrounded |

*For Gemini Vanilla:* Strong ≥ 52/60 generation; Adequate 42–51/60; Weak < 42/60.

---

## 5. Reporting Template

### A. Per-Pipeline Compliance Summary

```
Query: _______________________________________________
Pipeline: ____________________________________________

Compliance score: __ / 52
Automatic fail flags: Yes / No

Top missing compliance items:
  1.
  2.
  3.

Checklist verdict: Strong / Adequate / Weak
```

---

### B. Per-Pipeline Quality Summary

```
RETRIEVAL QUALITY
  Precision (relevance of retrieved chunks):     __ / 15
  Recall (coverage of full query scope):         __ / 15
  Temporal Accuracy (in-period prioritisation):  __ / 10
  Retrieval subtotal:                            __ / 40

GENERATION QUALITY
  Factual Correctness:                           __ / 25
  Grounding & Source Use:                        __ / 15
  Completeness:                                  __ / 10
  Clarity & Format:                              __ /  5
  Practical Usefulness:                          __ /  5
  Generation subtotal:                           __ / 60

Quality total:                                   __ / 100

Main quality concerns:
  1.
  2.
  3.
```

---

### C. Cross-Pipeline Comparison Table

*(Complete Step 4.3 table above for the query under evaluation.)*

---

### D. One-Paragraph Rationale

Write one paragraph explaining: which pipeline performed best on this query and why; what the best pipeline did that the others did not; and what, if anything, even the best pipeline failed to do. Reference specific scores and criteria rather than making general statements.

---

## 6. Test Query Set

The following 12 queries are recommended for evaluation. They are designed to cover the four query types specified in the assignment brief: simple facts, deep context, ambiguous questions, and edge cases.

| # | Query | Type | Key Criterion Under Test |
|---|---|---|---|
| 1 | What was Bitcoin's 7-day ATM implied volatility on March 12, 2026? | Simple fact | Retrieval precision; factual correctness; temporal accuracy |
| 2 | What were net ETF inflows for Bitcoin in the week of March 10, 2026? | Simple fact | Exact-term retrieval (BM25); grounding |
| 3 | What is implied volatility and how is it used in crypto markets? | Definitional | Response format rule; completeness; source use |
| 4 | How did Bitcoin's implied volatility term structure evolve throughout March 2026? | Deep context | Retrieval recall (month scope); temporal re-prioritisation; completeness |
| 5 | What drove the divergence between realised and implied volatility for Ethereum in Q1 2026? | Deep context | Quarter-scope retrieval; analytical format; synthesis across sources |
| 6 | How did institutional positioning in Bitcoin options change after the ETF approval period? | Deep context | Multi-document synthesis; usefulness; cross-source reasoning |
| 7 | Compare Bitcoin and Ethereum implied volatility performance in March 2026. | Comparative | Both assets addressed; temporal precision; comparative format rule |
| 8 | Compare ETH and BTC performance in Q1 2026. | Comparative | Quarter-scope adaptive retrieval; both assets cited separately |
| 9 | How is crypto doing lately? | Ambiguous | Scope detection failure case; implicit time reference handling |
| 10 | Is volatility high? | Ambiguous | No asset, no timeframe; grounding vs hallucination test |
| 11 | What was Bitcoin's performance in April 2027? | Edge case | Out-of-corpus query; limitation handling; no fabrication |
| 12 | What is the average implied volatility across all March 2026 documents? | Edge case | Aggregation query; correct refusal; limitation acknowledgement |
