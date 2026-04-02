"""Smart Order Router scoring rules.

Composite score per venue =
    α · normalized_volume_share
  + β · normalized_depth
  − γ · normalized_adverse_selection

Higher score → better venue to route to.
Non-routable venues (FINRA) get score 0 regardless of metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from hft.analysis.venue_metrics import NON_ROUTABLE_VENUES


@dataclass(frozen=True)
class SORWeights:
    alpha_volume: float = 0.5
    beta_depth: float = 0.3
    gamma_adverse: float = 0.2


def _zscore(s: pl.Series) -> pl.Series:
    """Standardise a series: (x - mean) / std. Returns zeros if std is 0."""
    m = s.mean()
    sd = s.std()
    if sd is None or sd == 0:
        return pl.Series([0.0] * len(s))
    return (s - m) / sd


def compute_composite_score(
    metrics: pl.DataFrame,
    *,
    weights: SORWeights | None = None,
) -> pl.DataFrame:
    """Add a `composite_score` column. Higher = better routing target."""
    w = weights or SORWeights()
    routable = metrics.filter(pl.col("routable"))
    if routable.is_empty():
        return metrics.with_columns(pl.lit(0.0).alias("composite_score"))

    # Z-score the three feature columns over routable venues only,
    # then attach back to the full DataFrame.
    vol_z = _zscore(routable["volume_share_pct"])
    dep_z = _zscore(routable["avg_depth_at_best"])
    adv_z = _zscore(routable["adverse_selection_bps"])

    score_routable = (
        w.alpha_volume * vol_z + w.beta_depth * dep_z - w.gamma_adverse * adv_z
    )
    routable_with_score = routable.with_columns(score_routable.alias("composite_score"))
    non_routable = metrics.filter(~pl.col("routable")).with_columns(
        pl.lit(0.0).alias("composite_score")
    )
    return pl.concat([routable_with_score, non_routable], how="diagonal").sort(
        "composite_score", descending=True
    )


def naive_volume_allocation(
    metrics: pl.DataFrame,
    *,
    top_k: int = 5,
    min_volume_pct: float = 0.0,
) -> dict[str, float]:
    """Allocate by volume share among the top-k routable venues. Sums to 1."""
    routable = (
        metrics.filter(pl.col("routable"))
        .filter(pl.col("volume_share_pct") >= min_volume_pct)
        .sort("volume_share_pct", descending=True)
        .head(top_k)
    )
    if routable.is_empty():
        return {}
    total = float(routable["volume_share_pct"].sum())
    return {row["Exchange"]: float(row["volume_share_pct"]) / total
            for row in routable.iter_rows(named=True)}


def sor_score_allocation(
    metrics_with_score: pl.DataFrame,
    *,
    top_k: int = 5,
    min_volume_pct: float = 3.0,    # require ≥3% daily volume to be eligible
) -> dict[str, float]:
    """Allocate by softmax of composite_score among the top-k routable venues
    that meet a minimum volume threshold.

    The min_volume_pct filter prevents tiny venues with noisy z-scores
    (small sample → spurious outlier metrics) from being over-allocated.
    """
    import math
    routable = (
        metrics_with_score.filter(pl.col("routable"))
        .filter(pl.col("volume_share_pct") >= min_volume_pct)
        .sort("composite_score", descending=True)
        .head(top_k)
    )
    if routable.is_empty():
        return {}
    scores = [float(s) for s in routable["composite_score"].to_list()]
    max_s = max(scores)
    exps = [math.exp(s - max_s) for s in scores]
    total = sum(exps)
    return {row["Exchange"]: exps[i] / total
            for i, row in enumerate(routable.iter_rows(named=True))}
