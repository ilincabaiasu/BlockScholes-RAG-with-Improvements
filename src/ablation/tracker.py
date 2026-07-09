"""Performance tracker — accumulates judged ablation runs into an Excel workbook
and computes per-component and per-group marginal contributions.

The workbook (``data/ablation/ablation_tracker.xlsx``) has four sheets and is
appended to on every runner invocation, so performance builds up over time:

  * ``Runs``       — one row per judged (batch, config, query) run (raw data).
  * ``Configs``    — per-config means across all accumulated runs.
  * ``Components`` — per-enhancement marginal contribution (FULL − no_<flag>),
                     paired per query within each batch, then averaged.
  * ``Groups``     — per-group joint contribution (FULL − no_grp_<group>).

Contributions are computed *paired*: for each query a config and FULL both ran
on, the delta is FULL_q − ablated_q. Pairing removes per-question difficulty as
a confounder, which is what makes the "is it worth it?" signal precise.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.ablation import config as cfgmod
from src.utils.logger import get_logger

_logger = get_logger(__name__)

TRACKER_PATH = Path("data/ablation/ablation_tracker.xlsx")

# Live-Q&A tracker: every question asked in the app is judged across all three
# pipelines and accumulated here, separate from the controlled CLI ablation.
QA_TRACKER_PATH = Path("data/ablation/qa_tracker.xlsx")

_QA_COLUMNS = [
    "batch_id", "question", "pipeline", "enhanced_config",
    "quality", "retrieval_pct", "generation",
    "factual_correctness", "grounding", "completeness", "clarity", "usefulness",
    "latency_ms", "fail_flags",
]

_PIPELINE_ORDER = {"vanilla": 0, "gemini": 0, "baseline": 1, "enhanced": 2}

# Column order for the raw Runs sheet.
_RUN_COLUMNS = [
    "batch_id", "config", "disabled", "query_id", "query",
    "quality", "retrieval_pct", "generation",
    "factual_correctness", "grounding", "completeness", "clarity", "usefulness",
    "latency_ms", "fail_flags",
]


def records_to_rows(records, batch_id: str) -> list[dict]:
    """Flatten judged RunRecords into Runs-sheet rows (skips errors/unjudged)."""
    rows: list[dict] = []
    for r in records:
        if r.error or not r.judge:
            continue
        j = r.judge
        m = r.metrics or {}
        retr_max = j.get("retrieval_max") or 0
        rows.append({
            "batch_id": batch_id,
            "config": r.config_name,
            "disabled": ",".join(r.disabled),
            "query_id": r.query_id,
            "query": r.query,
            "quality": j["quality_total_normalised"],
            "retrieval_pct": round(100.0 * j["retrieval_total"] / retr_max, 1) if retr_max else None,
            "generation": j["generation_total"],
            "factual_correctness": j["factual_correctness"],
            "grounding": j["grounding"],
            "completeness": j["completeness"],
            "clarity": j["clarity"],
            "usefulness": j["usefulness"],
            "latency_ms": (m.get("total_latency_ms") or 0.0),
            "fail_flags": len(m.get("auto_fail_flags") or []),
        })
    return rows


def _summarise_configs(runs: pd.DataFrame) -> pd.DataFrame:
    """Per-config means across all accumulated runs."""
    g = runs.groupby("config", as_index=False).agg(
        n_runs=("quality", "size"),
        mean_quality=("quality", "mean"),
        mean_generation=("generation", "mean"),
        mean_retrieval_pct=("retrieval_pct", "mean"),
        mean_latency_s=("latency_ms", lambda s: s.mean() / 1000.0),
        total_fail_flags=("fail_flags", "sum"),
    )
    for col in ("mean_quality", "mean_generation", "mean_retrieval_pct", "mean_latency_s"):
        g[col] = g[col].round(2)
    # FULL first, then NONE last, others alphabetical between.
    g["_order"] = g["config"].map(lambda c: 0 if c == "full" else (2 if c == "none" else 1))
    return g.sort_values(["_order", "config"]).drop(columns="_order").reset_index(drop=True)


def _paired_contribution(runs: pd.DataFrame, ablated_config: str) -> tuple[float, float, int] | None:
    """Mean (Δquality, Δlatency_s, n_pairs) of FULL − ablated_config.

    Paired per (batch_id, query_id): only queries where both FULL and the
    ablated config ran in the same batch contribute. Returns None if no pairs.
    """
    full = runs[runs["config"] == "full"]
    abl = runs[runs["config"] == ablated_config]
    if full.empty or abl.empty:
        return None

    keys = ["batch_id", "query_id"]
    full_g = full.groupby(keys, as_index=False).agg(q_full=("quality", "mean"), lat_full=("latency_ms", "mean"))
    abl_g = abl.groupby(keys, as_index=False).agg(q_abl=("quality", "mean"), lat_abl=("latency_ms", "mean"))
    merged = full_g.merge(abl_g, on=keys, how="inner")
    if merged.empty:
        return None

    dq = (merged["q_full"] - merged["q_abl"]).mean()
    dlat = ((merged["lat_full"] - merged["lat_abl"]) / 1000.0).mean()
    return round(float(dq), 2), round(float(dlat), 2), int(len(merged))


def _contributions(runs: pd.DataFrame, level: str) -> pd.DataFrame:
    """Marginal contributions for level in {'component', 'group'}."""
    if level == "component":
        items = [(f"no_{flag}", flag, cfgmod.group_of(flag) or "") for flag in cfgmod._ABLATABLE]
    else:
        items = [(f"no_grp_{g}", g, g) for g in cfgmod.GROUPS]

    out: list[dict] = []
    for cfg_name, key, group in items:
        res = _paired_contribution(runs, cfg_name)
        if res is None:
            continue
        dq, dlat, n = res
        # Value ratio: quality points bought per second of added latency.
        # Only meaningful when the component actually adds latency.
        value = round(dq / dlat, 2) if dlat > 0.05 else None
        row = {
            ("component" if level == "component" else "group"): key,
            "delta_quality": dq,
            "latency_cost_s": dlat,
            "quality_per_sec": value,
            "n_pairs": n,
        }
        if level == "component":
            row["group"] = group
        out.append(row)

    df = pd.DataFrame(out)
    if not df.empty:
        df = df.sort_values("delta_quality", ascending=False).reset_index(drop=True)
    return df


def update_tracker(records, batch_id: str) -> dict[str, pd.DataFrame]:
    """Append a batch of judged records and rewrite all derived sheets.

    Returns the four DataFrames keyed by sheet name.
    """
    rows = records_to_rows(records, batch_id)
    new = pd.DataFrame(rows, columns=_RUN_COLUMNS)
    if new.empty:
        _logger.warning(
            "tracker_no_judged_rows",
            extra={"batch_id": batch_id, "hint": "judge produced no scores — is the judge model reachable?"},
        )

    if TRACKER_PATH.exists():
        try:
            old = pd.read_excel(TRACKER_PATH, sheet_name="Runs")
            runs = pd.concat([old, new], ignore_index=True)
        except Exception as exc:  # corrupt/locked file — start fresh but warn
            _logger.warning("tracker_read_failed", extra={"error": str(exc)})
            runs = new
    else:
        runs = new

    sheets = {
        "Runs": runs,
        "Configs": _summarise_configs(runs),
        "Components": _contributions(runs, "component"),
        "Groups": _contributions(runs, "group"),
    }

    TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(TRACKER_PATH, engine="openpyxl") as xl:
        for name, df in sheets.items():
            (df if not df.empty else pd.DataFrame({"(no data yet)": []})).to_excel(
                xl, sheet_name=name, index=False
            )

    _logger.info(
        "tracker_updated",
        extra={"batch_id": batch_id, "new_rows": len(new), "total_rows": len(runs)},
    )
    return sheets


def _summarise_pipelines(runs: pd.DataFrame) -> pd.DataFrame:
    """Per-pipeline means across all judged Q&A runs."""
    g = runs.groupby("pipeline", as_index=False).agg(
        n_questions=("quality", "size"),
        mean_quality=("quality", "mean"),
        mean_retrieval_pct=("retrieval_pct", "mean"),
        mean_grounding=("grounding", "mean"),
        mean_factual=("factual_correctness", "mean"),
        mean_latency_s=("latency_ms", lambda s: s.mean() / 1000.0),
    )
    for col in ("mean_quality", "mean_retrieval_pct", "mean_grounding", "mean_factual", "mean_latency_s"):
        g[col] = g[col].round(2)
    g["_order"] = g["pipeline"].map(lambda p: _PIPELINE_ORDER.get(p, 9))
    return g.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def _summarise_enhanced_configs(runs: pd.DataFrame) -> pd.DataFrame:
    """Per-configuration means for the enhanced pipeline (which toggles were on)."""
    enh = runs[runs["pipeline"] == "enhanced"]
    if enh.empty:
        return pd.DataFrame()
    g = enh.groupby("enhanced_config", as_index=False).agg(
        n_questions=("quality", "size"),
        mean_quality=("quality", "mean"),
        mean_latency_s=("latency_ms", lambda s: s.mean() / 1000.0),
    )
    g["mean_quality"] = g["mean_quality"].round(2)
    g["mean_latency_s"] = g["mean_latency_s"].round(2)
    return g.sort_values("mean_quality", ascending=False).reset_index(drop=True)


_supabase_client_cache = None


def _get_supabase():
    """Return a cached Supabase client when credentials are available, else None."""
    global _supabase_client_cache
    if _supabase_client_cache is not None:
        return _supabase_client_cache
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        return None
    try:
        from supabase import create_client
        _supabase_client_cache = create_client(url, key)
        return _supabase_client_cache
    except Exception as exc:
        _logger.warning("supabase_init_failed", extra={"error": str(exc)})
        return None


def _load_qa_supabase(client) -> dict[str, pd.DataFrame] | None:
    try:
        result = client.table("qa_runs").select("*").order("id").execute()
    except Exception as exc:
        _logger.warning("supabase_load_failed", extra={"error": str(exc)})
        return None
    if not result.data:
        return None
    runs = pd.DataFrame(result.data)
    keep = [c for c in _QA_COLUMNS if c in runs.columns]
    if not keep or "pipeline" not in keep:
        return None
    runs = runs[keep]
    if runs.empty:
        return None
    return {
        "Runs": runs,
        "Pipelines": _summarise_pipelines(runs),
        "Enhanced_Configs": _summarise_enhanced_configs(runs),
    }


def _append_qa_supabase(client, rows: list[dict]) -> dict[str, pd.DataFrame]:
    try:
        client.table("qa_runs").insert(rows).execute()
        _logger.info("supabase_qa_inserted", extra={"n_rows": len(rows)})
    except Exception as exc:
        _logger.warning("supabase_insert_failed", extra={"error": str(exc)})
    return _load_qa_supabase(client) or {}


def append_qa(rows: list[dict]) -> dict[str, pd.DataFrame]:
    """Append judged Q&A rows. Uses Supabase when credentials are set, else local Excel."""
    client = _get_supabase()
    if client is not None:
        return _append_qa_supabase(client, rows)

    # --- local Excel fallback (dev without Supabase) ---
    new = pd.DataFrame(rows, columns=_QA_COLUMNS)
    if new.empty:
        return {}
    if QA_TRACKER_PATH.exists():
        try:
            old = pd.read_excel(QA_TRACKER_PATH, sheet_name="Runs")
            runs = pd.concat([old, new], ignore_index=True)
        except Exception as exc:
            _logger.warning("qa_tracker_read_failed", extra={"error": str(exc)})
            runs = new
    else:
        runs = new
    sheets = {
        "Runs": runs,
        "Pipelines": _summarise_pipelines(runs),
        "Enhanced_Configs": _summarise_enhanced_configs(runs),
    }
    QA_TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(QA_TRACKER_PATH, engine="openpyxl") as xl:
        for name, df in sheets.items():
            (df if not df.empty else pd.DataFrame({"(no data yet)": []})).to_excel(
                xl, sheet_name=name, index=False
            )
    _logger.info("qa_tracker_updated", extra={"new_rows": len(new), "total_rows": len(runs)})
    return sheets


def load_qa() -> dict[str, pd.DataFrame] | None:
    """Read live-Q&A history. Uses Supabase when credentials are set, else local Excel."""
    client = _get_supabase()
    if client is not None:
        return _load_qa_supabase(client)

    # --- local Excel fallback ---
    if not QA_TRACKER_PATH.exists():
        return None
    try:
        sheets = {
            name: pd.read_excel(QA_TRACKER_PATH, sheet_name=name)
            for name in ("Runs", "Pipelines", "Enhanced_Configs")
        }
    except Exception as exc:
        _logger.warning("qa_tracker_load_failed", extra={"error": str(exc)})
        return None
    runs = sheets.get("Runs")
    if runs is None or runs.empty or "pipeline" not in runs.columns:
        return None
    return sheets


def load_tracker() -> dict[str, pd.DataFrame] | None:
    """Read the tracker workbook, or None if it has no usable judged data yet."""
    if not TRACKER_PATH.exists():
        return None
    try:
        sheets = {
            name: pd.read_excel(TRACKER_PATH, sheet_name=name)
            for name in ("Runs", "Configs", "Components", "Groups")
        }
    except Exception as exc:
        _logger.warning("tracker_load_failed", extra={"error": str(exc)})
        return None
    runs = sheets.get("Runs")
    # A placeholder workbook (written when no run was judged) has no "config"
    # column / zero rows — treat that as "no data yet".
    if runs is None or runs.empty or "config" not in runs.columns:
        return None
    return sheets
