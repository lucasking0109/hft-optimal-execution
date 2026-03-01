"""Futures TAQ loader and L2 order book reconstruction.

The fut_taq files include per-price-level updates with explicit aggressor
direction (TRADE AGRESSOR ON BUY/SELL). reconstruct_book replays the updates
into a top-N snapshot at any timestamp without silent gap-filling.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from hft.data.loaders import (
    AVAILABLE_DATES,
    DATA_ROOT,
    PARQUET_CACHE_ROOT,
    DataIntegrityError,
    DiskSpaceError,
    check_disk_space,
)

FUT_TAQ_SCHEMA: dict[str, pl.DataType] = {
    "UTCDate": pl.Int64,
    "UTCTime": pl.Utf8,
    "LocalDate": pl.Int64,
    "LocalTime": pl.Utf8,
    "Ticker": pl.Categorical,
    "SecurityID": pl.Int64,
    "TypeMask": pl.Int64,
    "Type": pl.Categorical,
    "Price": pl.Float64,
    "Quantity": pl.Int64,
    "Orders": pl.Int64,
    "Flags": pl.Int64,
}

# NQ contract roots for January 2020 — Mar/Jun/Sep/Dec 2020 + Mar 2021
KNOWN_CONTRACTS: tuple[str, ...] = ("NQH0", "NQM0", "NQU0", "NQZ0", "NQH1")


def _fut_taq_csv_path(contract: str, date: str) -> Path:
    """Path to raw futures TAQ CSV.

    Layout: data/raw_link/fut_taq/{YYYY}/{YYYYMMDD}/{root}/{contract}.csv
    The root is "NQ" for the NASDAQ-100 family.
    """
    if len(date) != 8 or not date.isdigit():
        raise ValueError(f"date must be YYYYMMDD, got {date!r}")
    year = date[:4]
    root = contract[:2].upper()
    return (
        DATA_ROOT
        / "fut_taq"
        / year
        / date
        / root
        / f"{contract.upper()}.csv"
    )


def _fut_taq_parquet_path(contract: str, date: str) -> Path:
    return PARQUET_CACHE_ROOT / "fut_taq" / date / f"{contract.upper()}.parquet"


def _ensure_fut_cached(contract: str, date: str) -> Path:
    parquet = _fut_taq_parquet_path(contract, date)
    if parquet.exists():
        return parquet

    csv_path = _fut_taq_csv_path(contract, date)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Futures CSV missing: {csv_path}. "
            f"Known contracts: {KNOWN_CONTRACTS}. "
            f"Decide: spelling, mount, or pick valid contract."
        )

    csv_size_gb = csv_path.stat().st_size / 1e9
    check_disk_space(required_gb=max(0.3, csv_size_gb * 1.0))
    parquet.parent.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv(csv_path, schema_overrides=FUT_TAQ_SCHEMA, low_memory=False)
    df.write_parquet(parquet, compression="zstd", compression_level=3)
    return parquet


def load_fut_taq(contract: str, date: str, columns: list[str] | None = None) -> pl.DataFrame:
    """Load one futures contract × one day of tick data.

    Columns:
        UTCDate, UTCTime, LocalDate, LocalTime, Ticker, SecurityID, TypeMask,
        Type, Price, Quantity, Orders, Flags

    Type values include:
        QUOTE BID / QUOTE SELL
        TRADE / TRADE AGRESSOR ON BUY / TRADE AGRESSOR ON SELL
        SETTLEMENT PRICE / OPENING PRICE / OPEN INTEREST / EMPTY BOOK FINAL
    """
    if date not in AVAILABLE_DATES:
        raise ValueError(f"date {date} not in {AVAILABLE_DATES}")

    parquet = _ensure_fut_cached(contract, date)
    df = pl.read_parquet(parquet, columns=columns)

    if df.is_empty():
        raise DataIntegrityError(f"{contract} {date}: empty futures DataFrame")
    if "Price" in df.columns and df["Price"].null_count() > 0:
        raise DataIntegrityError(
            f"{contract} {date}: null Price values present. Decide how to handle."
        )
    return df


def _ts_to_nanos(ts: str) -> int:
    """Convert a 15-character HHMMSSnnnnnnnnn timestamp string to nanoseconds-of-day.

    Example: "120000123456789" → 12*3600s + 0m + 0s + 123456789 ns.
    """
    if len(ts) != 15:
        raise ValueError(f"expected 15-char HHMMSSnnnnnnnnn, got {ts!r}")
    h = int(ts[:2])
    m = int(ts[2:4])
    s = int(ts[4:6])
    ns = int(ts[6:])
    return ((h * 3600 + m * 60 + s) * 1_000_000_000) + ns


def reconstruct_book(
    df: pl.DataFrame,
    *,
    snapshot_time: str,
    depth: int = 5,
    time_column: str = "LocalTime",
) -> dict:
    """Reconstruct a top-N snapshot of the futures order book at snapshot_time.

    Replays per-price-level QUOTE BID / QUOTE SELL updates in time order. The
    last value at each price level wins; quantity 0 (or missing) means the level
    cleared.

    Args:
        df: futures TAQ DataFrame from load_fut_taq().
        snapshot_time: 15-char HHMMSSnnnnnnnnn or 6-char HHMMSS (treated as
            HHMMSS000000000).
        depth: number of price levels per side to return.
        time_column: which timestamp column to filter on (default LocalTime).

    Returns:
        {"timestamp": str, "bids": [(price, qty, orders), ...],
         "asks": [(price, qty, orders), ...]}

        Sides are sorted bids descending, asks ascending. Length ≤ depth.

    Raises:
        DataIntegrityError if the book is empty up to the requested time.
    """
    if len(snapshot_time) == 6:
        snapshot_time = snapshot_time + "000000000"
    target_ns = _ts_to_nanos(snapshot_time)

    if time_column not in df.columns:
        raise ValueError(f"{time_column} missing; columns are {df.columns}")

    # Convert the column lazily into nanoseconds-of-day for filtering
    timed = df.with_columns(
        pl.col(time_column).map_elements(_ts_to_nanos, return_dtype=pl.Int64).alias("_ns")
    ).filter(pl.col("_ns") <= target_ns)

    if timed.is_empty():
        raise DataIntegrityError(
            f"No quotes at or before {snapshot_time}. Earlier in session — try later snapshot."
        )

    # Latest quantity per (Type, Price) wins (sort ascending then group_by + last)
    quotes = (
        timed.filter(pl.col("Type").is_in(["QUOTE BID", "QUOTE SELL"]))
        .sort("_ns")
        .group_by(["Type", "Price"], maintain_order=False)
        .agg(
            pl.col("Quantity").last().alias("Quantity"),
            pl.col("Orders").last().alias("Orders"),
        )
        .filter(pl.col("Quantity") > 0)
    )

    bids = (
        quotes.filter(pl.col("Type") == "QUOTE BID")
        .sort("Price", descending=True)
        .head(depth)
    )
    asks = (
        quotes.filter(pl.col("Type") == "QUOTE SELL")
        .sort("Price", descending=False)
        .head(depth)
    )

    bid_list = list(zip(bids["Price"].to_list(), bids["Quantity"].to_list(), bids["Orders"].to_list()))
    ask_list = list(zip(asks["Price"].to_list(), asks["Quantity"].to_list(), asks["Orders"].to_list()))

    # Report crossed book honestly. Don't auto-clean — that would be a silent fallback.
    # Crossed book on this dataset is expected because: the feed is event-driven without
    # a full opening snapshot, and individual price levels can become stale when their
    # resting orders are filled via TRADE events without a corresponding QUOTE update.
    # Phase 1 EDA will determine the right reconstruction approach (snapshot bootstrapping,
    # stale-level TTL, or explicit EMPTY BOOK FINAL handling).
    is_crossed = bool(bid_list and ask_list and bid_list[0][0] >= ask_list[0][0])

    return {
        "timestamp": snapshot_time,
        "bids": bid_list,
        "asks": ask_list,
        "is_crossed": is_crossed,
    }
