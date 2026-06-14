"""Ablation configuration — feature flags for the enhanced pipeline.

Each flag turns off one query-time enhancement. The ``FULL`` config (every
flag on) reproduces the production enhanced pipeline exactly; this is the
invariant the pipeline refactor preserves.

Dependency notes (why some flags are coupled, see ABLATION.md):
  * ``hybrid`` off  → BM25 is skipped, so RRF degenerates to dense-only ranking.
    There is no separate "rrf" flag: RRF is meaningful only when both arms run.
  * ``rerank`` off  → the RRF/dense-sorted candidate list is used directly.
  * ``scope_adaptive`` off → retrieval breadth falls back to the static
    settings defaults instead of scaling with the detected temporal scope.
  * ``temporal_reprioritize`` reads the target period extracted during scope
    detection; it is a no-op for queries with no explicit period regardless.
  * ``parent_fetch`` off → context uses child-chunk text only (no hierarchical
    expansion). Hierarchical *chunking* itself is index-time and cannot be
    ablated here without re-indexing — see ABLATION.md.
"""

from __future__ import annotations

from dataclasses import dataclass, replace, fields


@dataclass(frozen=True)
class AblationConfig:
    """Which enhancements are active for a single pipeline run."""

    name: str = "full"

    classify: bool = True             # query classification → prompt + temperature
    rewrite: bool = True              # multi-query decomposition
    scope_adaptive: bool = True       # scale top_k / token budget to temporal scope
    hybrid: bool = True               # dense + BM25 (+ RRF); off → dense only
    rerank: bool = True               # Cohere cross-encoder reranking
    temporal_reprioritize: bool = True  # move in-period chunks to the front
    diversity_cap: bool = True        # cap chunks per source document
    parent_fetch: bool = True         # expand child chunks to parent context
    visual: bool = True               # render + interpret visual pages

    def disabled(self) -> list[str]:
        """Names of the flags that are turned off in this config."""
        return [
            f.name
            for f in fields(self)
            if f.name != "name" and getattr(self, f.name) is False
        ]


# All enhancements on — equivalent to the production enhanced pipeline.
FULL = AblationConfig(name="full")

# All query-time enhancements off — dense retrieval + vanilla generation.
# (This approximates the baseline pipeline but over the same hierarchical index.)
NONE = AblationConfig(
    name="none",
    classify=False,
    rewrite=False,
    scope_adaptive=False,
    hybrid=False,
    rerank=False,
    temporal_reprioritize=False,
    diversity_cap=False,
    parent_fetch=False,
    visual=False,
)


# Flags eligible for leave-one-out ablation (every togglable enhancement).
_ABLATABLE = [
    "classify",
    "rewrite",
    "scope_adaptive",
    "hybrid",
    "rerank",
    "temporal_reprioritize",
    "diversity_cap",
    "parent_fetch",
    "visual",
]


# Semantic groups for leave-one-group-out ablation. Each maps to a concrete
# engineering decision and bundles components that are *coupled*, so disabling
# the whole group captures interaction effects that single-flag LOO misses
# (e.g. RRF is meaningless unless both hybrid arms run; temporal reprioritise
# reads the scope/period that scope_adaptive sizes retrieval around).
GROUPS: dict[str, list[str]] = {
    "query_understanding": ["classify", "rewrite"],
    "hybrid_retrieval": ["hybrid", "rerank"],
    "temporal_targeting": ["scope_adaptive", "temporal_reprioritize"],
    "context_assembly": ["parent_fetch", "diversity_cap"],
    "visual": ["visual"],
}

# Human-readable labels for reports/charts.
GROUP_LABELS: dict[str, str] = {
    "query_understanding": "Query understanding",
    "hybrid_retrieval": "Hybrid retrieval & rerank",
    "temporal_targeting": "Temporal targeting",
    "context_assembly": "Context assembly",
    "visual": "Visual",
}


def group_of(flag: str) -> str | None:
    """Return the group name a flag belongs to, or None if ungrouped."""
    for group, flags in GROUPS.items():
        if flag in flags:
            return group
    return None


def leave_one_out() -> list[AblationConfig]:
    """One config per enhancement, with only that enhancement disabled.

    Isolates each component's marginal contribution by removing it from the
    otherwise-full pipeline (one-factor-at-a-time).
    """
    configs = []
    for flag in _ABLATABLE:
        configs.append(replace(FULL, name=f"no_{flag}", **{flag: False}))
    return configs


def leave_one_group_out() -> list[AblationConfig]:
    """One config per group, with every flag in that group disabled.

    Measures the *joint* marginal contribution of a coupled cluster of
    enhancements (vs FULL), which is more honest than summing per-flag LOO
    deltas when the components interact.
    """
    configs = []
    for group, flags in GROUPS.items():
        off = {flag: False for flag in flags}
        configs.append(replace(FULL, name=f"no_grp_{group}", **off))
    return configs


def default_suite() -> list[AblationConfig]:
    """The standard ablation sweep: FULL, every leave-one-out, and NONE."""
    return [FULL, *leave_one_out(), NONE]


def full_suite() -> list[AblationConfig]:
    """The complete sweep: FULL, per-component LOO, per-group LOGO, and NONE.

    De-duplicates single-member groups (e.g. ``visual``) whose LOGO config is
    identical to the matching LOO config.
    """
    seen: set[str] = set()
    suite: list[AblationConfig] = []
    for cfg in [FULL, *leave_one_out(), *leave_one_group_out(), NONE]:
        key = ",".join(sorted(cfg.disabled()))
        if key in seen:
            continue
        seen.add(key)
        suite.append(cfg)
    return suite
