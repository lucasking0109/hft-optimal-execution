"""Market VWAP computation.

Market VWAP = sum(price * qty) / sum(qty) over all market trades in a window.
Used as a benchmark for execution quality.
"""

from __future__ import annotations

import polars as pl


def compute_market_vwap(
    df: pl.DataFrame,
    *,
    start_ns: int | None = None,
    end_ns: int | None = None,
    use_nb_only: bool = False,
) -> float:
    """Compute volume-weighted average trade price over an optional time window.

    Args:
        df: equity TAQ DataFrame (must already have ns_of_day column OR
            we'll derive it from Timestamp).
        start_ns, end_ns: optional inclusive window (in nanoseconds-of-day).
        use_nb_only: if True, restrict to TRADE NB (NBBO trades). Default False.

    Returns:
        VWAP price as float.

    Raises:
        ValueError if no trades in window (NO silent fallback to mid or 0).
    """
    trades = df.filter(
        pl.col("EventType").is_in(["TRADE", "TRADE NB"]) if not use_nb_only
        else pl.col("EventType") == "TRADE NB"
    )
    if "ns_of_day" not in trades.columns:
        from hft.data.timeparse import add_eq_ns_of_day
        trades = add_eq_ns_of_day(trades)
    if start_ns is not None:
        trades = trades.filter(pl.col("ns_of_day") >= start_ns)
    if end_ns is not None:
        trades = trades.filter(pl.col("ns_of_day") <= end_ns)
    if trades.is_empty():
        window = f"[{start_ns}, {end_ns}]" if (start_ns or end_ns) else "all-day"
        raise ValueError(
            f"No trades in window {window}. Cannot compute VWAP. "
            f"Check window bounds or whether RTH filter dropped everything."
        )
    total_dollar = (trades["Price"] * trades["Quantity"]).sum()
    total_qty = trades["Quantity"].sum()
    if total_qty <= 0:
        raise ValueError(f"Trade volume is {total_qty}, cannot divide.")
    return float(total_dollar / total_qty)


def compute_volume_profile(
    df: pl.DataFrame,
    *,
    bin_minutes: int = 5,
    rth_only: bool = True,
) -> pl.DataFrame:
    """Per-bucket trade volume for the day.

    Returns DataFrame[bucket_min, volume]. Used by VWAP-following strategy
    as a forecast of market volume distribution.

    ⚠️  Computed from the input `df` — if `df` is the **current day's** TAQ
    this introduces look-ahead bias when used by a strategy. For
    leakage-free backtests use `compute_lookback_volume_profile()` below.
    """
    from hft.data.timeparse import add_eq_minute_bucket, filter_rth
    trades = df.filter(pl.col("EventType").is_in(["TRADE", "TRADE NB"]))
    if rth_only:
        trades = filter_rth(trades, src="Timestamp")
    bucketed = add_eq_minute_bucket(trades, bin_minutes=bin_minutes,
                                     src="Timestamp", dst="bucket_min")
    return (
        bucketed.group_by("bucket_min")
        .agg(pl.col("Quantity").sum().alias("volume"))
        .sort("bucket_min")
    )


def compute_lookback_volume_profile(
    ticker: str,
    *,
    lookback_dates: list[str],
    bin_minutes: int = 5,
    cache_root: "Path | None" = None,
) -> pl.DataFrame:
    """Cross-day rolling-average per-bucket volume profile (no leakage).

    Walks each date in `lookback_dates`, computes per-day volume profile,
    then averages per-bucket volume across all loaded days.

    Args:
        ticker: stock symbol (e.g. "AAPL").
        lookback_dates: explicit list of dates to average across. The
            evaluation date MUST be excluded by the caller — this function
            does not know which day you're evaluating.
        bin_minutes: bucket granularity (default 5 minutes).
        cache_root: override path to processed parquet cache.

    Returns:
        DataFrame[bucket_min, volume] — same schema as
        `compute_volume_profile()` so strategies don't need to change.

    Raises:
        RuntimeError if no parquet from `lookback_dates` is loadable
        (NO Silent Fallback — caller picks how to handle missing data).
    """
    from pathlib import Path

    if not lookback_dates:
        raise ValueError(
            "compute_lookback_volume_profile: lookback_dates is empty. "
            "Must provide at least one historical date to average over."
        )

    project_root = Path(__file__).resolve().parents[3]
    root = cache_root or (project_root / "data" / "processed" / "eq_taq")
    if not root.exists():
        raise RuntimeError(
            f"compute_lookback_volume_profile: cache root {root} not found"
        )

    per_day_profiles: list[pl.DataFrame] = []
    for date in lookback_dates:
        parquet = root / date / f"{ticker}.parquet"
        if not parquet.exists():
            continue
        try:
            df = pl.read_parquet(parquet)
        except Exception:
            continue
        try:
            prof = compute_volume_profile(df, bin_minutes=bin_minutes, rth_only=True)
        except Exception:
            continue
        per_day_profiles.append(prof)

    if not per_day_profiles:
        raise RuntimeError(
            f"compute_lookback_volume_profile: no parquet loadable for "
            f"{ticker} across {lookback_dates}"
        )

    # Stack and average per bucket_min.
    stacked = pl.concat(per_day_profiles, how="vertical")
    avg = (
        stacked.group_by("bucket_min")
        .agg(pl.col("volume").mean().alias("volume"))
        .sort("bucket_min")
    )
    return avg
