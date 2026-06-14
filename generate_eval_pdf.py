"""Generate EVAL_RUBRIC.pdf — streamlined single-layer rubric."""

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, HRFlowable, PageBreak,
    PageTemplate, Paragraph, Spacer, Table, TableStyle,
)

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
BLUE_DARK  = colors.HexColor("#1F4E79")
BLUE_MID   = colors.HexColor("#2E74B5")
BLUE_LIGHT = colors.HexColor("#C5D9F1")
BLUE_FAINT = colors.HexColor("#EBF3FB")
GREY_LINE  = colors.HexColor("#BFBFBF")
BLACK      = colors.HexColor("#1A1A1A")
WHITE      = colors.white

W, H   = A4
MARGIN = 2.2 * cm

# ---------------------------------------------------------------------------
# Paragraph styles
# ---------------------------------------------------------------------------
def _s(name, **kw):
    return ParagraphStyle(name, **kw)

TITLE    = _s("T",   fontName="Helvetica-Bold",   fontSize=22, textColor=BLUE_DARK,
               alignment=TA_CENTER, leading=28, spaceAfter=10)
SUBTITLE = _s("ST",  fontName="Helvetica",         fontSize=11, textColor=BLUE_MID,
               alignment=TA_CENTER, leading=16, spaceAfter=6)
BYLINE   = _s("BL",  fontName="Helvetica-Oblique", fontSize=9,  textColor=colors.HexColor("#595959"),
               alignment=TA_CENTER, leading=13, spaceAfter=18)

SEC      = _s("S",   fontName="Helvetica-Bold",   fontSize=13, textColor=BLUE_MID,
               spaceBefore=14, spaceAfter=4)
SUBSEC   = _s("SS",  fontName="Helvetica-Bold",   fontSize=10.5, textColor=BLUE_MID,
               spaceBefore=9, spaceAfter=3)
SUBSUBSEC= _s("SSS", fontName="Helvetica-Bold",   fontSize=9.5,  textColor=BLUE_DARK,
               spaceBefore=6, spaceAfter=2)

BODY     = _s("B",   fontName="Helvetica",         fontSize=9,  textColor=BLACK,
               leading=13, spaceAfter=4, alignment=TA_JUSTIFY)
BULLET   = _s("BU",  fontName="Helvetica",         fontSize=9,  textColor=BLACK,
               leading=13, spaceAfter=2, leftIndent=12)
NOTE     = _s("N",   fontName="Helvetica-Oblique", fontSize=8.5,textColor=colors.HexColor("#595959"),
               leading=11, spaceAfter=3, alignment=TA_JUSTIFY)

TH       = _s("TH",  fontName="Helvetica-Bold",   fontSize=8.5, textColor=BLUE_DARK,  alignment=TA_LEFT,   leading=11)
TD       = _s("TD",  fontName="Helvetica",         fontSize=8.5, textColor=BLACK,      alignment=TA_LEFT,   leading=11, wordWrap='LTR')
TD_B     = _s("TDB", fontName="Helvetica-Bold",   fontSize=8.5, textColor=BLACK,      alignment=TA_LEFT,   leading=11)
TD_C     = _s("TDC", fontName="Helvetica",         fontSize=8.5, textColor=BLACK,      alignment=TA_CENTER, leading=11)
TD_PT    = _s("TPT", fontName="Helvetica-Bold",   fontSize=9,   textColor=BLUE_MID,   alignment=TA_CENTER, leading=11)
TD_CRIT  = _s("TCR", fontName="Helvetica-Bold",   fontSize=8.5, textColor=BLUE_MID,   alignment=TA_LEFT,   leading=11, wordWrap='LTR')
TD_WT    = _s("TWT", fontName="Helvetica-Bold",   fontSize=9,   textColor=BLUE_MID,   alignment=TA_CENTER, leading=11)
TD_SM    = _s("TSM", fontName="Helvetica",         fontSize=7.8, textColor=BLACK,      alignment=TA_LEFT,   leading=10.5, wordWrap='LTR')
TH_SM    = _s("THS", fontName="Helvetica-Bold",   fontSize=8,   textColor=BLUE_DARK,  alignment=TA_LEFT,   leading=11)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def p(text, style=BODY):   return Paragraph(text, style)
def sp(h=4):               return Spacer(1, h)
def hr():                  return HRFlowable(width="100%", thickness=0.5, color=GREY_LINE, spaceAfter=6, spaceBefore=2)
def sec(t):                return p(t, SEC)
def subsec(t):             return p(t, SUBSEC)
def subsubsec(t):          return p(t, SUBSUBSEC)
def bullet(t):             return p(f"• {t}", BULLET)
def note(t):               return p(t, NOTE)

BASE_STYLE = [
    ("BACKGROUND",     (0, 0), (-1, 0),  BLUE_LIGHT),
    ("TEXTCOLOR",      (0, 0), (-1, 0),  BLUE_DARK),
    ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
    ("FONTSIZE",       (0, 0), (-1, 0),  8.5),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, BLUE_FAINT]),
    ("GRID",           (0, 0), (-1, -1), 0.4, GREY_LINE),
    ("VALIGN",         (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING",     (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
    ("LEFTPADDING",    (0, 0), (-1, -1), 5),
    ("RIGHTPADDING",   (0, 0), (-1, -1), 5),
]

def tbl(data, widths, extra=None):
    s = TableStyle(BASE_STYLE + (extra or []))
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(s)
    return t

U = W - 2 * MARGIN   # usable width

# ---------------------------------------------------------------------------
# Specific table builders
# ---------------------------------------------------------------------------
def two_col(rows, h1="Score Range", h2="Interpretation"):
    data = [[p(h1, TH), p(h2, TH)]] + [[p(a, TD_B), p(b, TD)] for a, b in rows]
    return tbl(data, [U*0.25, U*0.75])

def scoring_band_table(rows):
    data = [[p("Score", TH), p("Performance Level", TH)]] + \
           [[p(s, TD_B), p(d, TD)] for s, d in rows]
    return tbl(data, [U*0.13, U*0.87])

def rubric_table(rows):
    """Main 6-column quality rubric table."""
    headers = ["Criterion", "Pts.", "What It Means", "Why It Matters",
               "Strong Performance", "Weak Performance"]
    data = [[p(h, TH_SM) for h in headers]]
    for crit, pts, means, why, strong, weak in rows:
        data.append([
            p(crit,   TD_CRIT),
            p(pts,    TD_WT),
            p(means,  TD_SM),
            p(why,    TD_SM),
            p(strong, TD_SM),
            p(weak,   TD_SM),
        ])
    cw = [U*0.13, U*0.05, U*0.165, U*0.165, U*0.245, U*0.245]
    extra = [("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5)]
    return tbl(data, cw, extra)

def comparison_table():
    headers = ["", "Gemini Vanilla", "Baseline RAG", "Enhanced RAG"]
    rows = [
        ["Retrieval Precision",  "N/A",     "__ / 15", "__ / 15"],
        ["Retrieval Recall",     "N/A",     "__ / 15", "__ / 15"],
        ["Temporal Accuracy",    "N/A",     "__ / 10", "__ / 10"],
        ["Retrieval Total",      "0 / 40",  "__ / 40", "__ / 40"],
        ["Factual Correctness",  "__ / 25", "__ / 25", "__ / 25"],
        ["Grounding",            "__ / 20", "__ / 20", "__ / 20"],
        ["Completeness",         "__ / 10", "__ / 10", "__ / 10"],
        ["Clarity & Format",     "__ / 5",  "__ / 5",  "__ / 5"],
        ["Generation Total",     "__ / 60", "__ / 60", "__ / 60"],
        ["Quality Total",        "__ / 60", "__ / 100","__ / 100"],
        ["Key Strength",         "",        "",        ""],
        ["Key Weakness",         "",        "",        ""],
    ]
    data = [[p(h, TH) for h in headers]]
    shaded = {3, 8, 9}
    for i, row in enumerate(rows):
        data.append([p(row[0], TD_B)] + [p(c, TD_C) for c in row[1:]])
    extra = [("BACKGROUND", (0, r+1), (-1, r+1), BLUE_LIGHT) for r in shaded]
    extra += [("FONTNAME", (0, 4), (0, 9), "Helvetica-Bold"),
              ("FONTNAME", (0, 1), (0, 3), "Helvetica")]
    return tbl(data, [U*0.22, U*0.26, U*0.26, U*0.26], extra)

def query_table(rows):
    headers = ["#", "Query", "Type", "What It Isolates"]
    data = [[p(h, TH) for h in headers]]
    for row in rows:
        data.append([p(str(c), TD) for c in row])
    return tbl(data, [U*0.04, U*0.37, U*0.14, U*0.45])

def verdict_table(rows):
    data = [[p("Verdict", TH), p("Conditions", TH)]]
    for v, c in rows:
        data.append([p(v, TD_B), p(c, TD)])
    return tbl(data, [U*0.15, U*0.85])

# ---------------------------------------------------------------------------
# Page template
# ---------------------------------------------------------------------------
class Doc(BaseDocTemplate):
    def __init__(self, fn):
        super().__init__(fn, pagesize=A4,
                         leftMargin=MARGIN, rightMargin=MARGIN,
                         topMargin=MARGIN, bottomMargin=MARGIN)
        frame = Frame(MARGIN, MARGIN, U, H - 2*MARGIN,
                      topPadding=8, bottomPadding=6, id="main")
        self.addPageTemplates([PageTemplate(id="main", frames=[frame],
                                            onPage=self._footer)])

    @staticmethod
    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(GREY_LINE)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, MARGIN - 5*mm, W - MARGIN, MARGIN - 5*mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#595959"))
        canvas.drawCentredString(W/2, MARGIN - 10*mm,
                                  "Block Scholes RAG System — Evaluation Framework")
        canvas.drawRightString(W - MARGIN, MARGIN - 10*mm, str(doc.page))
        canvas.restoreState()

# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------
def build():
    story = []

    # ── Title ──────────────────────────────────────────────────────────────
    story += [
        sp(14),
        p("Evaluation Framework", TITLE),
        p("Block Scholes RAG System — Three-Pipeline Comparison", SUBTITLE),
        p("IBRAU9 Individual Assignment 2", BYLINE),
        hr(),
        sp(4),
    ]

    # ── 1. Introduction ────────────────────────────────────────────────────
    story += [
        sec("1. Introduction"),
        p(("This framework evaluates outputs from three pipelines of the Block Scholes Research Assistant "
           "across six representative test queries, each targeting a distinct system behaviour:"), BODY),
        sp(2),
        bullet("<b>Gemini Vanilla</b> — raw model knowledge, no retrieval"),
        bullet("<b>Baseline RAG</b> — dense-only retrieval, fixed 500-token chunks, minimal prompt"),
        bullet("<b>Enhanced RAG</b> — hybrid retrieval (dense + BM25), RRF fusion, Cohere cross-encoder "
               "reranking, parent-child chunking, adaptive scope detection, temporal re-prioritisation, "
               "and query-type-structured prompting"),
        sp(5),
        p(("The framework uses a single-layer <b>Quality Evaluation Rubric</b> (100 points) covering both "
           "retrieval and generation dimensions. Each pipeline is scored independently on the same query, "
           "enabling direct comparison. A short set of automatic fail flags precedes scoring to catch "
           "fundamental output failures before the rubric is applied."), BODY),
        sp(8),
    ]

    # ── 2. Automatic Fail Flags ────────────────────────────────────────────
    story += [
        sec("2. Automatic Fail Flags"),
        hr(),
        p(("Check these before scoring. If any apply, flag the output and note it in the rationale. "
           "Fail flags do not automatically zero the score, but they must be explained."), BODY),
        sp(4),
        bullet('Output opens with "the documents do not contain this information" and then provides that information — a direct contradiction'),
        bullet("A specific numerical figure is stated with no inline citation anywhere in the response"),
        bullet("A cited source title or date cannot be matched to any document in the retrieved set"),
        bullet("Response is fewer than three sentences for an analytical or comparative query"),
        bullet("For a time-scoped query: response discusses only events outside the stated period"),
        bullet("For an out-of-corpus query: response invents data rather than stating no results exist"),
        sp(10),
    ]

    # ── 3. Quality Evaluation Rubric ───────────────────────────────────────
    story += [
        sec("3. Quality Evaluation Rubric"),
        hr(),
        subsec("3.1 Purpose"),
        p(("Score each pipeline output on seven criteria totalling 100 points. Retrieval criteria "
           "are scored N/A (0 points counted) for Gemini Vanilla, which performs no retrieval — "
           "its effective maximum is 60 points on generation criteria only."), BODY),
        sp(4),
        tbl(
            [[p("Domain", TH), p("Criterion", TH), p("Points", TH)],
             [p("Retrieval", TD_B), p("Precision — relevance of retrieved chunks", TD), p("15", TD_C)],
             [p("",          TD),   p("Recall — coverage of the full query scope", TD),  p("15", TD_C)],
             [p("",          TD),   p("Temporal Accuracy — in-period sources prioritised", TD), p("10", TD_C)],
             [p("",          TD_B), p("Retrieval Subtotal", TD_B), p("40", TD_C)],
             [p("Generation",TD_B), p("Factual Correctness — claims accurate vs. sources", TD), p("25", TD_C)],
             [p("",          TD),   p("Grounding — every claim cited; no outside knowledge", TD), p("20", TD_C)],
             [p("",          TD),   p("Completeness — full query scope addressed", TD), p("10", TD_C)],
             [p("",          TD),   p("Clarity & Format — structure matches query type", TD), p("5",  TD_C)],
             [p("",          TD_B), p("Generation Subtotal", TD_B), p("60", TD_C)],
             [p("",          TD_B), p("Total", TD_B), p("100", TD_C)],
            ],
            [U*0.16, U*0.68, U*0.16],
            extra=[
                ("BACKGROUND", (0, 4), (-1, 4), BLUE_LIGHT),
                ("BACKGROUND", (0, 9), (-1, 9), BLUE_LIGHT),
                ("BACKGROUND", (0, 10),(-1,10), BLUE_LIGHT),
                ("FONTNAME",   (0, 4), (-1, 4), "Helvetica-Bold"),
                ("FONTNAME",   (0, 9), (-1, 9), "Helvetica-Bold"),
                ("FONTNAME",   (0,10), (-1,10), "Helvetica-Bold"),
                ("SPAN",       (0, 1), (0, 3)),
                ("SPAN",       (0, 5), (0, 8)),
            ]
        ),
        sp(10),
    ]

    # ── 3.2 Rubric table ──
    story += [
        subsec("3.2 Rubric Detail"),
        note("Retrieval criteria are scored N/A for Gemini Vanilla. Score N/A for Temporal Accuracy on queries with no time period."),
        sp(4),
        rubric_table([
            ("Retrieval\nPrecision", "15",
             "Of the chunks in the context window, the proportion that are actually relevant to the query. Penalises noise — chunks from the wrong asset, the wrong period, or a tangentially related concept.",
             "Low precision dilutes the context window. Irrelevant chunks displace relevant ones within the token budget, leaving the LLM with less evidence and increasing the risk of vague or hallucinated answers.",
             "Context window is predominantly relevant; every chunk plausibly contributes to answering the query; no clearly off-topic material",
             "Several chunks concern a different asset, time period, or concept; context window contains filler that does not contribute to the answer"),
            ("Retrieval\nRecall", "15",
             "Whether the retrieved set covers the full scope of the query. For month-scoped queries: sources spanning the full month. For multi-part queries: supporting documents for each sub-question.",
             "Low recall produces incomplete answers regardless of generation quality. If the retrieval stage misses half the relevant documents, the LLM cannot synthesise what it was never given.",
             "For time-scoped queries: sources span the full stated period; for multi-part queries: supporting documents present for each sub-question; no large coverage gaps",
             "Documents cluster on a single event within a broader period; one sub-question has no supporting sources; significant portions of the query scope are unrepresented"),
            ("Temporal\nAccuracy", "10",
             "For time-scoped queries: whether in-period sources appear before out-of-period sources in the context window. Tests the temporal re-prioritisation mechanism.",
             "Cross-encoders score semantic relevance, not temporal relevance. Without re-prioritisation, a semantically similar article from the wrong month can displace a temporally correct one, producing a misleading answer.",
             "In-period sources appear first; answer cites documents from the correct period; out-of-period sources appear only after in-period sources are exhausted",
             "Out-of-period sources appear ahead of in-period ones; answer cites articles from the wrong month or quarter; correct-period documents were retrieved but displaced"),
            ("Factual\nCorrectness", "25",
             "All specific claims — implied volatility figures, ETF flow volumes, spot prices, dates, market directions — are accurate relative to the cited source documents. Where data is absent the model says so explicitly.",
             "In financial research, an inaccurate answer is worse than no answer. A wrong volatility figure or misattributed ETF flow can directly mislead analytical conclusions.",
             "All figures traceable to cited sources; BTC and ETH never conflated; market directions accurate; where data is absent the model explicitly states this rather than estimating",
             "Hallucinated figures; wrong asset attributed to wrong date; market direction stated incorrectly; specific data presented as fact with no traceable origin in any retrieved document"),
            ("Grounding\n& Citations", "20",
             "Every factual claim is anchored to a specific retrieved document using the [Source: Title | Date] format. The model draws exclusively from retrieved context, not its parametric training memory.",
             "RAG exists to prevent hallucination and ground answers in verifiable sources. An uncited claim is unverifiable regardless of whether it happens to be correct.",
             "All factual claims carry inline citations; citation format consistent throughout; no outside knowledge introduced; sources cited in text match the retrieved set",
             "Claims appear without citation; model introduces figures or events not in the retrieved documents; citation format inconsistently applied or absent for key claims"),
            ("Completeness", "10",
             "The answer addresses the full scope of the query — all sub-questions, the full stated time period, and all assets in a comparative query. Gaps that are acknowledged score better than gaps that are silently omitted.",
             "Incomplete answers mislead by omission. A response covering only Bitcoin in a BTC-ETH comparison, or only one week of data for a full-month query, fails the user even if what it covers is accurate.",
             "Full stated period covered; all assets addressed in comparatives; every sub-question answered; when the corpus lacks data the model explicitly says so",
             "Only part of the period discussed; one asset missing from a comparative with no acknowledgement; multiple sub-questions absent; significant scope silently omitted"),
            ("Clarity\n& Format", "5",
             "Response structure matches the query type. Factual queries lead with the answer. Analytical queries show visible reasoning. Definitional queries explain and illustrate with data. Terminology is consistent.",
             "Format signals whether the model understood what was asked. A one-sentence response to a complex analytical question indicates the model did not engage at the appropriate depth.",
             "Format matches query type; analytical responses show structured reasoning; definitional responses illustrate with data; terminology consistent throughout",
             "One-sentence answer to an analytical query; no structure visible; contradictions in terminology; reader cannot follow the reasoning"),
        ]),
        sp(10),
    ]

    # ── 3.3 Scoring guides ──
    story += [subsec("3.3 Scoring Guides by Criterion"), sp(4)]

    guides = [
        ("Retrieval Precision — 15 points", [
            ("13–15", "Context window predominantly relevant; all chunks plausibly contribute to the answer"),
            ("9–12",  "Mostly relevant; a few off-topic chunks present but not dominant"),
            ("0–8",   "Significant proportion of context window is off-topic or from the wrong period"),
        ]),
        ("Retrieval Recall — 15 points", [
            ("13–15", "Full query scope covered; supporting sources present for every sub-question or time segment"),
            ("9–12",  "Partial coverage; one sub-question or part of the period underrepresented"),
            ("0–8",   "Significant gaps; major portions of the query scope have no supporting sources"),
        ]),
        ("Temporal Accuracy — 10 points", [
            ("9–10", "In-period sources appear first; answer cites correct-period documents throughout"),
            ("6–8",  "Mostly correct; one or two out-of-period sources appear ahead of in-period ones"),
            ("0–5",  "Out-of-period sources dominate the context window; answer cites the wrong period"),
        ]),
        ("Factual Correctness — 25 points", [
            ("22–25", "Fully accurate; all figures traceable to cited sources; no fabrication; absent data explicitly noted"),
            ("15–21", "Mostly accurate; minor non-critical errors that do not change the substance of the answer"),
            ("0–14",  "Material factual errors; hallucinated figures; claims that contradict their cited source"),
        ]),
        ("Grounding & Citations — 20 points", [
            ("18–20", "Every factual claim cited; format consistent throughout; no outside knowledge"),
            ("12–17", "Most claims cited; one or two ungrounded statements; format mostly consistent"),
            ("0–11",  "Frequent ungrounded claims; citation-free responses; outside knowledge used materially"),
        ]),
        ("Completeness — 10 points", [
            ("9–10", "Full query scope addressed; all sub-questions answered; gaps acknowledged explicitly"),
            ("6–8",  "Partial coverage; some sub-questions or part of the period unaddressed"),
            ("0–5",  "Significant scope missed; silent truncation; major sub-questions absent"),
        ]),
        ("Clarity & Format — 5 points", [
            ("5",   "Format appropriate to query type; reasoning easy to follow; terminology consistent"),
            ("3–4", "Readable but minor format mismatch or uneven depth"),
            ("0–2", "Format does not match query type; contradictions; reasoning not followable"),
        ]),
    ]

    for title, bands in guides:
        story += [subsubsec(title), scoring_band_table(bands), sp(6)]

    # ── 3.4 Score interpretation ──
    story += [
        sp(4),
        subsec("3.4 Score Interpretation"),
        two_col([
            ("88–100", "High-quality output — accurate, grounded, complete, and analytically useful"),
            ("72–87",  "Good output with meaningful gaps in at least one criterion"),
            ("55–71",  "Weak output — systematic issues in retrieval or generation"),
            ("Below 55","Not fit for research use"),
        ]),
        sp(6),
        note("For Gemini Vanilla the maximum is 60 points (generation only). "
             "Interpret its score out of 60: ≥ 52 strong, 42–51 adequate, < 42 weak."),
        sp(10),
        PageBreak(),
    ]

    # ── 4. Evaluation Workflow ─────────────────────────────────────────────
    story += [sec("4. Evaluation Workflow"), hr()]

    story += [
        subsec("4.1 Step 1 — Check Automatic Fail Flags"),
        p("Run through Section 2 before opening the rubric. Note any flags in the rationale field. Do not skip this step even if the answer looks superficially good.", BODY),
        sp(8),
        subsec("4.2 Step 2 — Score Each Criterion"),
        p("For each pipeline output, score every criterion using the scoring bands in Section 3.3. Add a one-sentence comment per criterion explaining the primary reason for the score.", BODY),
        sp(8),
        subsec("4.3 Step 3 — Complete the Cross-Pipeline Comparison Table"),
        p("Fill in the table below for each query. This makes the score differences visible at a glance.", BODY),
        sp(4),
        comparison_table(),
        sp(10),
        subsec("4.4 Step 4 — Form the Final Verdict"),
        verdict_table([
            ("Strong",   "No fail flags; Quality total ≥ 82/100; Factual Correctness ≥ 20/25"),
            ("Adequate", "No fail flags; Quality total 65–81/100; core requirements met with gaps"),
            ("Weak",     "Any fail flag; Quality total < 65/100; or Factual Correctness < 15/25"),
        ]),
        note("For Gemini Vanilla: Strong ≥ 50/60; Adequate 40–49/60; Weak < 40/60."),
        sp(10),
        PageBreak(),
    ]

    # ── 5. Reporting Template ──────────────────────────────────────────────
    story += [sec("5. Reporting Template"), hr()]

    story += [
        subsec("A. Per-Pipeline Score Summary"),
        tbl(
            [[p("Criterion", TH), p("Gemini Vanilla", TH), p("Baseline RAG", TH), p("Enhanced RAG", TH)],
             [p("Retrieval Precision",  TD_B), p("N/A",     TD_C), p("__ / 15", TD_C), p("__ / 15", TD_C)],
             [p("Retrieval Recall",     TD_B), p("N/A",     TD_C), p("__ / 15", TD_C), p("__ / 15", TD_C)],
             [p("Temporal Accuracy",    TD_B), p("N/A",     TD_C), p("__ / 10", TD_C), p("__ / 10", TD_C)],
             [p("Factual Correctness",  TD_B), p("__ / 25", TD_C), p("__ / 25", TD_C), p("__ / 25", TD_C)],
             [p("Grounding",            TD_B), p("__ / 20", TD_C), p("__ / 20", TD_C), p("__ / 20", TD_C)],
             [p("Completeness",         TD_B), p("__ / 10", TD_C), p("__ / 10", TD_C), p("__ / 10", TD_C)],
             [p("Clarity & Format",     TD_B), p("__ / 5",  TD_C), p("__ / 5",  TD_C), p("__ / 5",  TD_C)],
             [p("Total",                TD_B), p("__ / 60", TD_C), p("__ / 100",TD_C), p("__ / 100",TD_C)],
             [p("Fail Flags",           TD_B), p("Yes / No",TD_C), p("Yes / No",TD_C), p("Yes / No",TD_C)],
             [p("Verdict",              TD_B), p("",        TD_C), p("",        TD_C), p("",        TD_C)],
            ],
            [U*0.28, U*0.24, U*0.24, U*0.24],
            extra=[("BACKGROUND", (0, 8), (-1, 8), BLUE_LIGHT),
                   ("FONTNAME",   (0, 8), (-1, 8), "Helvetica-Bold")]
        ),
        sp(10),
        subsec("B. One-Paragraph Rationale"),
        p(("Write one paragraph per query explaining: which pipeline performed best and why; "
           "what the best pipeline did that the others did not; and what, if anything, even the best "
           "pipeline failed to do. Reference specific scores rather than making general statements."), BODY),
        sp(10),
        PageBreak(),
    ]

    # ── 6. Test Query Set ──────────────────────────────────────────────────
    story += [sec("6. Test Query Set"), hr()]

    story += [
        p(("Six queries covering the four categories specified in the assignment brief. "
           "Each query is chosen to isolate a specific system behaviour, making score differences "
           "between pipelines interpretable rather than incidental."), BODY),
        sp(6),
        query_table([
            ("1", "What was Bitcoin's most recent 7-day ATM implied volatility?",
             "Simple fact",
             "Recency-anchored retrieval precision; factual correctness; exact-term recall (BM25 vs dense)"),
            ("2", "What is implied volatility and how is it used in crypto markets?",
             "Definitional",
             "Definitional format rule; completeness of explanation; grounding vs. parametric knowledge"),
            ("3", "How did Bitcoin's implied volatility term structure evolve throughout March 2026?",
             "Deep context",
             "Month-scope retrieval recall; temporal re-prioritisation; analytical format; synthesis across sources"),
            ("4", "Compare Bitcoin and Ethereum implied volatility performance in April 2026.",
             "Comparative",
             "Both assets cited separately; temporal precision; comparative prompt format; completeness"),
            ("5", "How is crypto doing lately?",
             "Ambiguous",
             "Implicit time reference — scope detection failure case; grounding vs. hallucination test"),
            ("6", "What was Bitcoin's performance in April 2027?",
             "Edge case",
             "Out-of-corpus query; limitation handling; absence of fabrication"),
        ]),
        sp(10),
        subsec("6.1 What Each Query Tests Across Pipelines"),
        tbl(
            [[p("Query", TH), p("Gemini Vanilla weakness", TH), p("Baseline RAG weakness", TH), p("Enhanced RAG advantage", TH)],
             [p("Q1 — Simple fact", TD_B),
              p("May hallucinate a plausible but wrong figure", TD),
              p("Fixed chunks may split the data point across chunk boundary", TD),
              p("BM25 catches exact term '7D ATM IV'; recency-weighted retrieval surfaces latest report", TD),
              ],
             [p("Q2 — Definitional", TD_B),
              p("Answers from training memory without any grounding", TD),
              p("May give one-sentence answer if prompt does not enforce format", TD),
              p("Structured prompt forces definition + mechanics + illustrated example", TD)],
             [p("Q3 — Deep context", TD_B),
              p("No temporal grounding; answer reflects training data, not March 2026", TD),
              p("Static top-k may under-retrieve for a full-month query", TD),
              p("Month-scope detection scales retrieval; temporal re-prioritisation ensures correct period", TD)],
             [p("Q4 — Comparative", TD_B),
              p("May address only one asset or conflate BTC/ETH data; temporal framing speculative", TD),
              p("One-size-fits-all prompt may not enforce separate treatment of each asset", TD),
              p("Comparative prompt template enforces separate sections with individual citations; April 2026 corpus coverage available", TD)],
             [p("Q5 — Ambiguous", TD_B),
              p("Answers based on training knowledge with no grounding", TD),
              p("Retrieves recent documents but cannot anchor to 'lately' — scope detection does not fire", TD),
              p("Same limitation as baseline — shared edge case; tests whether both cite sources or hallucinate", TD)],
             [p("Q6 — Edge case", TD_B),
              p("High risk of hallucinating plausible-sounding but fabricated data", TD),
              p("Should return no sources and state absence; risk of generic answer instead", TD),
              p("Should explicitly state no in-corpus data for April 2027; tests limitation handling", TD)],
            ],
            [U*0.14, U*0.28, U*0.28, U*0.30]
        ),
    ]

    out = "/Users/ilincabaiasu/Desktop/RAG/block-scholes-rag/EVAL_RUBRIC.pdf"
    Doc(out).build(story)
    print(f"PDF written to {out}")

if __name__ == "__main__":
    build()
