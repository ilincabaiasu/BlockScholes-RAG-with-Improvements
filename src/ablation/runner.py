"""Ablation runner — executes the config sweep over the test queries.

For every (config, query) pair it runs the enhanced pipeline with the given
ablation flags, computes programmatic metrics, and (optionally) scores the
answer with the LLM judge. It then aggregates per-config means and writes a
JSON dump plus a Markdown report, including a leave-one-out delta table showing
each component's marginal contribution to quality.

Usage:
    python -m src.ablation.runner                  # full sweep, with judge
    python -m src.ablation.runner --configs loo    # only leave-one-out configs
    python -m src.ablation.runner --queries 1 4 7  # subset of queries
    python -m src.ablation.runner --no-judge       # programmatic metrics only
    python -m src.ablation.runner --concurrency 2  # limit parallel queries
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean

from src.ablation import config as cfgmod
from src.ablation.config import AblationConfig
from src.ablation.judge import JudgeScores, judge_answer
from src.ablation.metrics import Metrics, compute_metrics
from src.ablation.queries import TEST_QUERIES, EvalQuery
from src.ablation import tracker
from src.pipelines import enhanced_pipeline
from src.utils.logger import get_logger

_logger = get_logger(__name__)

_OUT_DIR = Path("data/ablation")


@dataclass
class RunRecord:
    config_name: str
    disabled: list[str]
    query_id: int
    query: str
    metrics: dict | None = None
    judge: dict | None = None
    error: str | None = None


async def run_one(cfg: AblationConfig, eq: EvalQuery, use_judge: bool) -> RunRecord:
    """Run a single (config, query) pair and score it."""
    try:
        result = await enhanced_pipeline.run(eq.query, config=cfg)
    except Exception as exc:  # pragma: no cover - network/runtime failures
        _logger.warning(
            "ablation_run_failed",
            extra={"config": cfg.name, "query_id": eq.id, "error": str(exc)},
        )
        return RunRecord(cfg.name, cfg.disabled(), eq.id, eq.query, error=str(exc))

    metrics = compute_metrics(result, eq)
    judge_dict = None
    if use_judge:
        try:
            scores = await judge_answer(result, eq)
            judge_dict = scores.to_dict()
        except Exception as exc:  # pragma: no cover
            _logger.warning(
                "ablation_judge_failed",
                extra={"config": cfg.name, "query_id": eq.id, "error": str(exc)},
            )

    return RunRecord(
        config_name=cfg.name,
        disabled=cfg.disabled(),
        query_id=eq.id,
        query=eq.query,
        metrics=metrics.to_dict(),
        judge=judge_dict,
    )


async def run_suite(
    configs: list[AblationConfig],
    queries: list[EvalQuery],
    use_judge: bool,
    concurrency: int,
) -> list[RunRecord]:
    """Run every config over every query with bounded concurrency."""
    sem = asyncio.Semaphore(concurrency)
    records: list[RunRecord] = []

    async def _guarded(cfg: AblationConfig, eq: EvalQuery) -> RunRecord:
        async with sem:
            _logger.info("ablation_run_start", extra={"config": cfg.name, "query_id": eq.id})
            return await run_one(cfg, eq, use_judge)

    for cfg in configs:
        # Queries within a config run concurrently; configs run sequentially so
        # progress is readable and rate-limit pressure stays bounded.
        tasks = [_guarded(cfg, eq) for eq in queries]
        cfg_records = await asyncio.gather(*tasks)
        records.extend(cfg_records)
        done = [r for r in cfg_records if r.error is None]
        _logger.info(
            "ablation_config_done",
            extra={"config": cfg.name, "ok": len(done), "total": len(cfg_records)},
        )
        print(f"  ✓ {cfg.name}: {len(done)}/{len(cfg_records)} queries ok")

    return records


# ---------------------------------------------------------------------------
# Aggregation + reporting
# ---------------------------------------------------------------------------

@dataclass
class ConfigSummary:
    config_name: str
    disabled: list[str]
    n_ok: int
    n_failed: int
    mean_quality: float | None
    mean_generation: float | None
    mean_retrieval_pct: float | None
    mean_latency_ms: float | None
    total_fail_flags: int
    mean_distinct_docs: float | None


def _safe_mean(values: list[float]) -> float | None:
    return round(mean(values), 1) if values else None


def summarise(records: list[RunRecord]) -> list[ConfigSummary]:
    """Aggregate run records into one summary per config."""
    by_cfg: dict[str, list[RunRecord]] = {}
    order: list[str] = []
    for r in records:
        if r.config_name not in by_cfg:
            by_cfg[r.config_name] = []
            order.append(r.config_name)
        by_cfg[r.config_name].append(r)

    summaries: list[ConfigSummary] = []
    for name in order:
        rs = by_cfg[name]
        ok = [r for r in rs if r.error is None]
        quality = [r.judge["quality_total_normalised"] for r in ok if r.judge]
        generation = [r.judge["generation_total"] for r in ok if r.judge]
        retr_pct = [
            100.0 * r.judge["retrieval_total"] / r.judge["retrieval_max"]
            for r in ok
            if r.judge and r.judge["retrieval_max"]
        ]
        latency = [r.metrics["total_latency_ms"] for r in ok if r.metrics]
        distinct = [r.metrics["n_distinct_docs"] for r in ok if r.metrics]
        fail_flags = sum(len(r.metrics["auto_fail_flags"]) for r in ok if r.metrics)

        summaries.append(
            ConfigSummary(
                config_name=name,
                disabled=rs[0].disabled,
                n_ok=len(ok),
                n_failed=len(rs) - len(ok),
                mean_quality=_safe_mean(quality),
                mean_generation=_safe_mean(generation),
                mean_retrieval_pct=_safe_mean(retr_pct),
                mean_latency_ms=_safe_mean(latency),
                total_fail_flags=fail_flags,
                mean_distinct_docs=_safe_mean(distinct),
            )
        )
    return summaries


def build_report(summaries: list[ConfigSummary], use_judge: bool) -> str:
    """Render a Markdown report with the per-config table and LOO deltas."""
    lines: list[str] = ["# Ablation Results", ""]
    lines.append(f"_Generated: {datetime.utcnow().isoformat()}Z_")
    lines.append("")

    # Main table
    lines.append("## Per-config means")
    lines.append("")
    lines.append(
        "| Config | Disabled | OK | Quality/100 | Gen/60 | Retr% | Fail flags | Latency ms |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in summaries:
        disabled = ", ".join(s.disabled) if s.disabled else "—"
        lines.append(
            f"| `{s.config_name}` | {disabled} | {s.n_ok} | "
            f"{s.mean_quality if s.mean_quality is not None else 'n/a'} | "
            f"{s.mean_generation if s.mean_generation is not None else 'n/a'} | "
            f"{s.mean_retrieval_pct if s.mean_retrieval_pct is not None else 'n/a'} | "
            f"{s.total_fail_flags} | "
            f"{s.mean_latency_ms if s.mean_latency_ms is not None else 'n/a'} |"
        )
    lines.append("")

    # Contribution deltas vs FULL — components (LOO) and groups (LOGO) kept
    # in separate tables. "no_grp_*" are group configs, "no_*" (the rest) are
    # single-component configs.
    full = next((s for s in summaries if s.config_name == "full"), None)
    if full and full.mean_quality is not None and use_judge:

        def _delta_table(heading: str, configs: list[ConfigSummary], prefix: str) -> None:
            scored = [s for s in configs if s.mean_quality is not None]
            if not scored:
                return
            lines.append(heading)
            lines.append("")
            lines.append(
                "Positive Δ means removing it **hurt** quality — i.e. it contributed that "
                "many points. Negative means it slightly helped to remove it."
            )
            lines.append("")
            lines.append("| Removed | Δ Quality/100 | Δ Gen/60 | Δ Fail flags |")
            lines.append("|---|---|---|---|")
            for s in sorted(scored, key=lambda s: full.mean_quality - s.mean_quality, reverse=True):
                dq = round(full.mean_quality - s.mean_quality, 1)
                dg = (
                    round(full.mean_generation - s.mean_generation, 1)
                    if s.mean_generation is not None and full.mean_generation is not None
                    else "n/a"
                )
                dflags = s.total_fail_flags - full.total_fail_flags
                lines.append(f"| `{s.config_name[len(prefix):]}` | {dq:+} | {dg} | {dflags:+} |")
            lines.append("")

        loo = [s for s in summaries if s.config_name.startswith("no_") and not s.config_name.startswith("no_grp_")]
        logo = [s for s in summaries if s.config_name.startswith("no_grp_")]
        _delta_table("## Component contribution (FULL − leave-one-out)", loo, "no_")
        _delta_table("## Group contribution (FULL − leave-one-group-out)", logo, "no_grp_")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Chunking strategy (fixed vs hierarchical) is **not** ablated here — it is "
        "index-time and requires re-embedding the corpus into a second collection."
    )
    lines.append(
        "- Retrieval precision/recall are judged from source titles+dates only, so "
        "treat retrieval scores as relative signals across configs, not absolutes."
    )
    if not use_judge:
        lines.append("- Run with `--no-judge`: quality/generation/retrieval columns are blank.")
    lines.append("")
    return "\n".join(lines)


def _resolve_configs(name: str) -> list[AblationConfig]:
    if name == "all":
        return cfgmod.default_suite()
    if name == "everything":
        return cfgmod.full_suite()
    if name == "loo":
        return cfgmod.leave_one_out()
    if name == "logo":
        return [cfgmod.FULL, *cfgmod.leave_one_group_out()]
    if name == "full":
        return [cfgmod.FULL]
    if name == "none":
        return [cfgmod.NONE]
    if name == "endpoints":
        return [cfgmod.FULL, cfgmod.NONE]
    raise ValueError(f"unknown --configs value: {name!r}")


async def _amain(args: argparse.Namespace) -> None:
    configs = _resolve_configs(args.configs)
    queries = (
        [q for q in TEST_QUERIES if q.id in set(args.queries)]
        if args.queries
        else TEST_QUERIES
    )
    use_judge = not args.no_judge

    print(
        f"Ablation: {len(configs)} configs × {len(queries)} queries "
        f"= {len(configs) * len(queries)} runs (judge={'on' if use_judge else 'off'})"
    )

    records = await run_suite(configs, queries, use_judge, args.concurrency)
    summaries = summarise(records)
    report = build_report(summaries, use_judge)

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    json_path = _OUT_DIR / f"results_{stamp}.json"
    md_path = _OUT_DIR / f"report_{stamp}.md"

    json_path.write_text(
        json.dumps(
            {
                "generated": datetime.utcnow().isoformat() + "Z",
                "records": [asdict(r) for r in records],
                "summaries": [asdict(s) for s in summaries],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    md_path.write_text(report, encoding="utf-8")

    # Append this batch to the cumulative Excel tracker (Runs / Configs /
    # Components / Groups), so per-component and per-group contributions build
    # up across runs and feed the in-app performance dashboard.
    tracker_paths = ""
    if use_judge:
        tracker.update_tracker(records, batch_id=stamp)
        tracker_paths = f"\n      {tracker.TRACKER_PATH}"

    print("\n" + report)
    print(f"\nWrote {json_path}\n      {md_path}{tracker_paths}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the enhanced-pipeline ablation sweep.")
    parser.add_argument(
        "--configs",
        default="all",
        help="all | everything | loo | logo | full | none | endpoints (default: all). "
             "'everything' = FULL + per-component LOO + per-group LOGO + NONE.",
    )
    parser.add_argument(
        "--queries",
        nargs="*",
        type=int,
        default=None,
        help="Query ids to run (default: all 12)",
    )
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM judge")
    parser.add_argument(
        "--concurrency", type=int, default=3, help="Max parallel queries per config"
    )
    args = parser.parse_args()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()