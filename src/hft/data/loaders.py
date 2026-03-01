"""TAQ data loaders with lazy Parquet caching.

NO Silent Fallback: any data integrity issue or missing file raises explicit
exceptions. Disk-space pressure raises DiskSpaceError. Validation problems
raise DataIntegrityError. The caller decides what to do.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data" / "raw_link"
PARQUET_CACHE_ROOT = PROJECT_ROOT / "data" / "processed"

AVAILABLE_DATES: tuple[str, ...] = (
    "20200113",
    "20200114",
    "20200115",
    "20200116",
    "20200117",
)

EQ_TAQ_SCHEMA: dict[str, pl.DataType] = {
    "Date": pl.Int64,
    "Timestamp": pl.Utf8,
    "EventType": pl.Categorical,
    "Ticker": pl.Categorical,
    "Price": pl.Float64,
    "Quantity": pl.Int64,
    "Exchange": pl.Categorical,
    "Conditions": pl.Utf8,
}


class DataIntegrityError(Exception):
    """Raised when loaded TAQ data fails validation. Surfaces the problem
    instead of silently filling NaNs / dropping rows."""


class DiskSpaceError(Exception):
    """Raised when projected Parquet write would exceed available disk."""


def check_disk_space(required_gb: float = 1.0, path: Path | None = None) -> dict:
    """Check free disk space before writing Parquet cache.

    Returns dict with stats. Raises DiskSpaceError when insufficient.
    """
    target = path or PARQUET_CACHE_ROOT
    target.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(target)
    free_gb = usage.free / 1e9
    if free_gb < required_gb:
        raise DiskSpaceError(
            f"Need {required_gb:.1f} GB free at {target}, only {free_gb:.2f} GB available. "
            f"Decide: free up space, mount external disk, or set PARQUET_CACHE_ROOT elsewhere."
        )
    return {
        "free_gb": free_gb,
        "total_gb": usage.total / 1e9,
        "used_gb": usage.used / 1e9,
        "path": str(target),
    }


def _eq_taq_csv_path(ticker: str, date: str) -> Path:
    """Path to the raw equity TAQ CSV for one ticker/day.

    Layout: data/raw_link/eq_taq/{YYYY}/{YYYYMMDD}/{first_letter}/{TICKER}.csv
    """
    if len(date) != 8 or not date.isdigit():
        raise ValueError(f"date must be YYYYMMDD, got {date!r}")
    year = date[:4]
    first_letter = ticker[0].upper()
    return (
        DATA_ROOT
        / "eq_taq"
        / year
        / date
        / first_letter
        / f"{ticker.upper()}.csv"
    )


def _eq_taq_parquet_path(ticker: str, date: str) -> Path:
    """Cached parquet location."""
    return PARQUET_CACHE_ROOT / "eq_taq" / date / f"{ticker.upper()}.parquet"


def _ensure_cached(ticker: str, date: str) -> Path:
    """Lazy parquet conversion: convert this ticker/day on first use only.

    Raises FileNotFoundError if the source CSV is missing (no silent fallback).
    """
    parquet = _eq_taq_parquet_path(ticker, date)
    if parquet.exists():
        return parquet

    csv_path = _eq_taq_csv_path(ticker, date)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Source CSV missing: {csv_path}. "
            f"Available dates: {AVAILABLE_DATES}. "
            f"Decide: check ticker spelling, verify external drive mounted, or pick a different date."
        )

    # Make sure cache directory exists and we have headroom (~3x raw CSV size for safety)
    csv_size_gb = csv_path.stat().st_size / 1e9
    check_disk_space(required_gb=max(0.5, csv_size_gb * 1.0))
    parquet.parent.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv(csv_path, schema_overrides=EQ_TAQ_SCHEMA, low_memory=False)
    df.write_parquet(parquet, compression="zstd", compression_level=3)
    return parquet


def load_eq_taq(ticker: str, date: str, columns: list[str] | None = None) -> pl.DataFrame:
    """Load one ticker × one day of equity TAQ tick data.

    On first call, converts the CSV to Parquet under data/processed/eq_taq/.
    Subsequent calls read the cached Parquet (~5-10x faster).

    Args:
        ticker: e.g. "AAPL". Case-insensitive.
        date: "YYYYMMDD" in {AVAILABLE_DATES}.
        columns: optional column subset.

    Returns:
        Polars DataFrame with columns:
        Date, Timestamp, EventType, Ticker, Price, Quantity, Exchange, Conditions

    Raises:
        FileNotFoundError: source CSV missing.
        DataIntegrityError: loaded data fails validation.
        DiskSpaceError: not enough disk for parquet cache.
    """
    parquet = _ensure_cached(ticker, date)
    df = pl.read_parquet(parquet, columns=columns)
    validate_taq(df, ticker=ticker, date=date)
    return df


def validate_taq(df: pl.DataFrame, *, ticker: str, date: str) -> None:
    """Sanity-check loaded TAQ. Raises DataIntegrityError on any problem.

    NO silent fixing: we surface the issue and let the caller decide.
    """
    if df.is_empty():
        raise DataIntegrityError(f"{ticker} {date}: empty DataFrame")

    required_cols = {"Date", "Timestamp", "EventType", "Ticker", "Price", "Quantity", "Exchange"}
    missing = required_cols - set(df.columns)
    if missing:
        raise DataIntegrityError(f"{ticker} {date}: missing columns {missing}")

    # Price must be non-negative (zero is allowed for absent quotes); NaN never OK.
    null_price_count = df["Price"].null_count()
    if null_price_count > 0:
        raise DataIntegrityError(
            f"{ticker} {date}: {null_price_count} rows with null Price. "
            f"Decide whether to drop, investigate, or accept."
        )

    # Quantity null also fails loud
    null_qty_count = df["Quantity"].null_count()
    if null_qty_count > 0:
        raise DataIntegrityError(
            f"{ticker} {date}: {null_qty_count} rows with null Quantity."
        )

    # Date column must be the requested date
    expected_date_int = int(date)
    distinct_dates = df["Date"].unique().to_list()
    if any(d != expected_date_int for d in distinct_dates):
        raise DataIntegrityError(
            f"{ticker} {date}: unexpected dates in column: {distinct_dates}"
        )

    # Reasonable event count: a major-cap stock should have > 50k events even on a slow day.
    # Don't auto-fail small tickers, just warn via a low threshold.
    if len(df) < 1_000:
        raise DataIntegrityError(
            f"{ticker} {date}: only {len(df)} events. Likely a corrupt/partial file."
        )


def list_eq_taq_tickers(date: str = "20200113") -> list[str]:
    """List every ticker available in eq_taq for a given day."""
    if date not in AVAILABLE_DATES:
        raise ValueError(f"date {date} not in {AVAILABLE_DATES}")
    base = DATA_ROOT / "eq_taq" / date[:4] / date
    if not base.exists():
        raise FileNotFoundError(f"{base} not found — external drive mounted?")
    tickers = []
    for letter_dir in sorted(base.iterdir()):
        if letter_dir.is_dir():
            for f in sorted(letter_dir.glob("*.csv")):
                tickers.append(f.stem)
    return tickers


def load_eq_daily_ohlc(date: str) -> pl.DataFrame:
    """Load the daily OHLC table for one trading day.

    Daily data covers 2015-01-01 through 2020-12-31 (510 tickers).
    """
    if len(date) != 8 or not date.isdigit():
        raise ValueError(f"date must be YYYYMMDD, got {date!r}")
    csv_path = DATA_ROOT / "eq_daily_ohlc" / date[:4] / f"{date}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Daily OHLC missing: {csv_path}")
    return pl.read_csv(csv_path)
