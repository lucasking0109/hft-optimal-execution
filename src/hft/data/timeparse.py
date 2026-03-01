"""Vectorised timestamp parsing for tick data.

Equity TAQ Timestamp:  "HH:MM:SS.nnnnnnnnn"  (e.g. "04:00:00.062779093")
Futures TAQ time:      "HHMMSSnnnnnnnnn"     (e.g. "153000123456789")

Both are nanoseconds-of-day. We add a polars-native helper so we can avoid
Python loops on 5M+ rows.
"""

from __future__ import annotations

import polars as pl

# Regular trading hours (US equities, ET): 09:30:00 - 16:00:00
RTH_START_NS = (9 * 3600 + 30 * 60) * 1_000_000_000
RTH_END_NS = 16 * 3600 * 1_000_000_000


def add_eq_ns_of_day(df: pl.DataFrame, src: str = "Timestamp", dst: str = "ns_of_day") -> pl.DataFrame:
    """Add a nanoseconds-of-day Int64 column derived from "HH:MM:SS.nnnnnnnnn"."""
    if src not in df.columns:
        raise ValueError(f"{src} column missing; have {df.columns}")

    return df.with_columns(
        (
            pl.col(src).str.slice(0, 2).cast(pl.Int64) * 3600 * 1_000_000_000
            + pl.col(src).str.slice(3, 2).cast(pl.Int64) * 60 * 1_000_000_000
            + pl.col(src).str.slice(6, 2).cast(pl.Int64) * 1_000_000_000
            + pl.col(src).str.slice(9, 9).cast(pl.Int64)
        ).alias(dst)
    )


def add_eq_seconds_of_day(df: pl.DataFrame, src: str = "Timestamp", dst: str = "sec_of_day") -> pl.DataFrame:
    """Add a seconds-of-day Float64 column. Used for plotly numeric x-axis."""
    if "ns_of_day" in df.columns:
        return df.with_columns((pl.col("ns_of_day") / 1e9).alias(dst))
    df = add_eq_ns_of_day(df, src=src)
    return df.with_columns((pl.col("ns_of_day") / 1e9).alias(dst))


def hour_tick_labels(start_sec: float, end_sec: float, every_seconds: int = 1800) -> tuple[list[float], list[str]]:
    """Generate (tickvals, ticktext) for an x-axis showing HH:MM marks."""
    first = int(start_sec // every_seconds) * every_seconds
    if first < start_sec:
        first += every_seconds
    tickvals = list(range(first, int(end_sec) + 1, every_seconds))
    ticktext = [f"{t//3600:02d}:{(t % 3600)//60:02d}" for t in tickvals]
    return tickvals, ticktext


def add_eq_minute_bucket(df: pl.DataFrame, *, bin_minutes: int = 5,
                         src: str = "Timestamp", dst: str = "bucket_min") -> pl.DataFrame:
    """Add a minute-of-day bucket integer (e.g. 9:30 with bin_minutes=5 → 570/5 = 114).

    Useful for grouping volume into intraday bars.
    """
    if src not in df.columns:
        raise ValueError(f"{src} column missing")
    return df.with_columns(
        (
            (
                pl.col(src).str.slice(0, 2).cast(pl.Int64) * 60
                + pl.col(src).str.slice(3, 2).cast(pl.Int64)
            )
            // bin_minutes
        ).alias(dst)
    )


def add_eq_minute_label(df: pl.DataFrame, *, src: str = "Timestamp", dst: str = "hhmm") -> pl.DataFrame:
    """Add a "HH:MM" string label column."""
    return df.with_columns(pl.col(src).str.slice(0, 5).alias(dst))


def filter_rth(df: pl.DataFrame, src: str = "Timestamp") -> pl.DataFrame:
    """Filter to Regular Trading Hours (09:30:00–16:00:00 ET)."""
    df = add_eq_ns_of_day(df, src=src, dst="_ns_of_day_tmp")
    return (
        df.filter((pl.col("_ns_of_day_tmp") >= RTH_START_NS) & (pl.col("_ns_of_day_tmp") < RTH_END_NS))
        .drop("_ns_of_day_tmp")
    )


def add_fut_ns_of_day(df: pl.DataFrame, src: str = "LocalTime", dst: str = "ns_of_day") -> pl.DataFrame:
    """Futures format is HHMMSSnnnnnnnnn (15 chars, no separators)."""
    if src not in df.columns:
        raise ValueError(f"{src} missing")
    return df.with_columns(
        (
            pl.col(src).str.slice(0, 2).cast(pl.Int64) * 3600 * 1_000_000_000
            + pl.col(src).str.slice(2, 2).cast(pl.Int64) * 60 * 1_000_000_000
            + pl.col(src).str.slice(4, 2).cast(pl.Int64) * 1_000_000_000
            + pl.col(src).str.slice(6, 9).cast(pl.Int64)
        ).alias(dst)
    )
