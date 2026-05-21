"""Almgren-Chriss market-impact parameter estimation.

Two parameters in the AC linear-impact model:
    - η (eta)   : temporary impact coefficient (resilient, depends on trading rate)
    - γ (gamma) : permanent impact coefficient (information leakage)

Estimation philosophy (per Phase 3 plan):
    - η: estimable from 5 days of tick. Identify large trade events, measure
      short-term mid drift, regress on size.
    - γ: NOT reliably estimable from 5 days (academic consensus). Instead we
      hold literature priors and *verify* whether 5-day regression matches.
      If it disagrees by > some factor, we surface the disagreement loudly
      (NO Silent Fallback) so the caller decides.

We work in **dimensionless units** for portability:
    - size_x_ADV  = trade_qty / average_daily_volume    (so 0.001 == 0.1% ADV)
    - impact_bps  = price drift in basis points
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from hft.analysis.nbbo_lookup import NBBOLookup
from hft.data.timeparse import add_eq_ns_of_day, filter_rth

# ---------------------------------------------------------------------------
# Literature priors (used as sanity check for γ)
# Sources (rough orders of magnitude from Almgren et al., Bouchaud et al.):
#   For US large-caps, permanent impact roughly 3–10 bps per 1% of ADV
#   Temporary impact roughly 5–20 bps per 1% of ADV traded over 1 minute
# Per-ticker calibrations are kept offline; downstream callers fall back to
# `_default` via dict.get().
# ---------------------------------------------------------------------------
GAMMA_PRIOR_BPS_PER_PCT_ADV: dict[str, float] = {
    "_default": 5.0,
}

# Temporary impact priors — we use these as the primary η downstream because
# 5 days of tick is too few to estimate η reliably with OLS (R² ≈ 0).
# Values follow Almgren et al. linear-impact range for US large caps.
ETA_PRIOR_BPS_PER_PCT_ADV: dict[str, float] = {
    "_default": 12.0,
}

# Reasonable eta range so we can flag absurd estimates
ETA_PRIOR_RANGE_BPS_PER_PCT_ADV: tuple[float, float] = (1.0, 50.0)


@dataclass
class EtaEstimate:
    """Dimensionless temporary-impact estimate.

    Units: (impact in bps) per (1% of ADV traded).

    `slope` is the regression coefficient; `intercept` should be ~0 if the
    linear model fits well.
    """
    eta_bps_per_pct_adv: float
    intercept_bps: float
    n_events: int
    r_squared: float
    impact_window_seconds: int
    percentile_threshold: float
    out_of_range: bool   # True if estimate is outside ETA_PRIOR_RANGE


@dataclass
class GammaCheck:
    """Comparison of our 5-day γ estimate against literature prior."""
    prior_bps_per_pct_adv: float
    estimated_bps_per_pct_adv: float
    n_events: int
    ratio: float                 # estimated / prior
    consistent: bool             # True iff ratio ∈ [0.25, 4.0]
    note: str


def estimate_eta(
    df: pl.DataFrame,
    *,
    adv_shares: float,
    percentile_threshold: float = 0.90,
    impact_window_seconds: int = 5,
    rth_only: bool = True,
    min_events: int = 30,
) -> EtaEstimate:
    """Estimate temporary impact η from large trade events.

    Algorithm:
        1. Filter to RTH trades.
        2. Determine large-trade threshold = qty quantile P_threshold.
        3. For each large trade event:
            mid_before = NBBO mid 1ms before
            mid_after  = NBBO mid impact_window_seconds after
            signed_impact_bps = sign × (mid_after / mid_before − 1) × 10000
                where sign = +1 for buy aggressor, −1 for sell
                (Lee-Ready inferred from trade-vs-mid_before)
            size_pct_adv = qty / adv_shares × 100
        4. OLS: signed_impact_bps = α + η · size_pct_adv

    Raises:
        ValueError if too few qualifying events (NO silent fallback to a
        bad estimate). Caller sees the failure.
    """
    if "ns_of_day" not in df.columns:
        df = add_eq_ns_of_day(df)
    if rth_only:
        df = filter_rth(df, src="Timestamp")

    trades = df.filter(pl.col("EventType").is_in(["TRADE", "TRADE NB"]))
    if trades.is_empty():
        raise ValueError("No trades after RTH filter; cannot estimate η")

    threshold = trades["Quantity"].quantile(percentile_threshold)
    large = trades.filter(pl.col("Quantity") >= threshold).sort("ns_of_day")

    lookup = NBBOLookup(df)
    horizon_ns = impact_window_seconds * 1_000_000_000

    sizes_pct_adv: list[float] = []
    impacts_bps: list[float] = []

    for row in large.iter_rows(named=True):
        ns = row["ns_of_day"]
        qty = row["Quantity"]
        trade_px = row["Price"]
        if qty is None or qty <= 0 or trade_px is None or trade_px <= 0:
            continue

        # Use mid 1ms before to avoid same-event NBBO artefacts
        mid_before = lookup.mid_at(ns - 1_000_000)
        mid_after = lookup.mid_at(ns + horizon_ns)
        if mid_before is None or mid_after is None or mid_before <= 0 or mid_after <= 0:
            continue

        # Lee-Ready sign: trade above mid → buy aggressor, below → sell
        if trade_px > mid_before * (1 + 1e-9):
            sign = +1
        elif trade_px < mid_before * (1 - 1e-9):
            sign = -1
        else:
            continue   # ambiguous; skip
        impact_bps = sign * (mid_after - mid_before) / mid_before * 10000

        size_pct = qty / adv_shares * 100
        sizes_pct_adv.append(size_pct)
        impacts_bps.append(impact_bps)

    if len(sizes_pct_adv) < min_events:
        raise ValueError(
            f"Only {len(sizes_pct_adv)} valid large-trade events; need ≥ {min_events}. "
            f"Lower percentile_threshold or extend impact_window_seconds."
        )

    x = np.array(sizes_pct_adv)
    y = np.array(impacts_bps)
    # OLS y = a + b*x
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    intercept, slope = coef[0], coef[1]
    y_pred = intercept + slope * x
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    out_of_range = not (
        ETA_PRIOR_RANGE_BPS_PER_PCT_ADV[0] <= abs(slope) <= ETA_PRIOR_RANGE_BPS_PER_PCT_ADV[1]
    )
    return EtaEstimate(
        eta_bps_per_pct_adv=float(slope),
        intercept_bps=float(intercept),
        n_events=len(sizes_pct_adv),
        r_squared=float(r_sq),
        impact_window_seconds=impact_window_seconds,
        percentile_threshold=percentile_threshold,
        out_of_range=out_of_range,
    )


def check_gamma_against_prior(
    df: pl.DataFrame,
    *,
    ticker: str,
    adv_shares: float,
    long_horizon_seconds: int = 600,
    percentile_threshold: float = 0.95,
    rth_only: bool = True,
) -> GammaCheck:
    """Estimate γ using *long-horizon* mid drift after large trades, compare
    to literature prior.

    γ in our convention = bps of permanent (post-resilience) drift per 1% ADV.

    Per the plan: 5 days of tick is too sparse to ESTIMATE γ reliably; the
    purpose of this check is to flag whether our 5-day measurement is in
    the same order of magnitude as the literature prior. Result is reported
    transparently in the calibration CSV."""
    if "ns_of_day" not in df.columns:
        df = add_eq_ns_of_day(df)
    if rth_only:
        df = filter_rth(df, src="Timestamp")

    trades = df.filter(pl.col("EventType").is_in(["TRADE", "TRADE NB"]))
    threshold = trades["Quantity"].quantile(percentile_threshold)
    large = trades.filter(pl.col("Quantity") >= threshold).sort("ns_of_day")
    lookup = NBBOLookup(df)
    horizon_ns = long_horizon_seconds * 1_000_000_000

    impacts_bps: list[float] = []
    sizes_pct_adv: list[float] = []
    for row in large.iter_rows(named=True):
        ns = row["ns_of_day"]
        qty = row["Quantity"]
        trade_px = row["Price"]
        mid_before = lookup.mid_at(ns - 1_000_000)
        mid_after = lookup.mid_at(ns + horizon_ns)
        if mid_before is None or mid_after is None or mid_before <= 0 or mid_after <= 0:
            continue
        if trade_px > mid_before * (1 + 1e-9):
            sign = +1
        elif trade_px < mid_before * (1 - 1e-9):
            sign = -1
        else:
            continue
        impacts_bps.append(sign * (mid_after - mid_before) / mid_before * 10000)
        sizes_pct_adv.append(qty / adv_shares * 100)

    prior = GAMMA_PRIOR_BPS_PER_PCT_ADV.get(ticker, GAMMA_PRIOR_BPS_PER_PCT_ADV["_default"])

    if len(sizes_pct_adv) < 20:
        return GammaCheck(
            prior_bps_per_pct_adv=prior,
            estimated_bps_per_pct_adv=float("nan"),
            n_events=len(sizes_pct_adv),
            ratio=float("nan"),
            consistent=False,
            note=f"Only {len(sizes_pct_adv)} events; not enough to estimate γ reliably (which is fine — 5 days too few). Using prior.",
        )

    x = np.array(sizes_pct_adv)
    y = np.array(impacts_bps)
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope = float(coef[1])
    ratio = abs(slope) / prior if prior > 0 else float("inf")
    consistent = 0.25 <= ratio <= 4.0
    note = (
        f"5-day regression slope={slope:.2f} bps/%ADV vs prior {prior:.2f}; "
        f"ratio={ratio:.2f} → "
        + ("consistent ✅" if consistent else "⚠️ disagreement (5 days too few, prior used downstream)")
    )
    return GammaCheck(
        prior_bps_per_pct_adv=prior,
        estimated_bps_per_pct_adv=slope,
        n_events=len(sizes_pct_adv),
        ratio=ratio,
        consistent=consistent,
        note=note,
    )


def average_daily_volume(daily_df: pl.DataFrame, ticker: str, *, lookback_days: int = 60) -> float:
    """Estimate ADV (in shares) from the daily OHLC table.

    Uses MarketHoursVolume column over the most recent `lookback_days` rows.
    """
    if "Ticker" not in daily_df.columns or "MarketHoursVolume" not in daily_df.columns:
        raise ValueError("daily_df must have Ticker and MarketHoursVolume columns")
    sub = daily_df.filter(pl.col("Ticker") == ticker)
    if sub.is_empty():
        raise ValueError(f"Ticker {ticker} not found in daily OHLC")
    if len(sub) > lookback_days:
        sub = sub.tail(lookback_days)
    return float(sub["MarketHoursVolume"].mean())


def adv_from_tick_files(ticker: str, dates: list[str]) -> float:
    """Compute ADV by summing TRADE NB quantities across the given days.

    Useful for tickers absent from the daily OHLC universe (e.g., TSLA in the
    Jan 2020 NDX-100 sample). NOT a silent fallback — caller chooses to use
    this only when daily lookup explicitly fails.
    """
    from hft.data import load_eq_taq
    from hft.data.timeparse import filter_rth

    totals: list[int] = []
    for date in dates:
        df = load_eq_taq(ticker, date)
        df = filter_rth(df, src="Timestamp")
        trades = df.filter(pl.col("EventType").is_in(["TRADE", "TRADE NB"]))
        totals.append(int(trades["Quantity"].sum()))
    if not totals:
        raise ValueError(f"No data found for {ticker} across {dates}")
    return sum(totals) / len(totals)


def average_daily_volume_from_history(
    ticker: str,
    *,
    history_dir,
    lookback_days: int = 60,
) -> float:
    """Walk multiple daily OHLC CSVs to compute multi-year ADV."""
    from hft.data import DATA_ROOT
    base = history_dir or (DATA_ROOT / "eq_daily_ohlc")
    rows: list[float] = []
    for year_dir in sorted(base.iterdir(), reverse=True):
        if not year_dir.is_dir():
            continue
        for csv_path in sorted(year_dir.glob("*.csv"), reverse=True):
            df = pl.read_csv(csv_path, columns=["Ticker", "MarketHoursVolume"])
            sub = df.filter(pl.col("Ticker") == ticker)
            if not sub.is_empty():
                rows.append(float(sub["MarketHoursVolume"][0]))
            if len(rows) >= lookback_days:
                break
        if len(rows) >= lookback_days:
            break
    if not rows:
        raise ValueError(f"No daily volume history found for {ticker}")
    return sum(rows) / len(rows)
