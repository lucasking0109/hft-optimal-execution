"""Phase 0 sanity tests. NO silent fallback — every test must verify a
guarantee we depend on later. Failures here mean the data pipeline is broken
and downstream research is unreliable."""

from __future__ import annotations

import time

import polars as pl
import pytest

from hft.data import (
    AVAILABLE_DATES,
    DATA_ROOT,
    DataIntegrityError,
    check_disk_space,
    list_eq_taq_tickers,
    load_eq_daily_ohlc,
    load_eq_taq,
)
from hft.data.nbbo import build_nbbo
from hft.data.orderbook import load_fut_taq, reconstruct_book


def test_disk_space_check_returns_stats():
    stats = check_disk_space(required_gb=0.001)
    assert stats["free_gb"] > 0
    assert stats["total_gb"] >= stats["free_gb"]


def test_disk_space_check_raises_when_insufficient():
    from hft.data.loaders import DiskSpaceError

    with pytest.raises(DiskSpaceError):
        check_disk_space(required_gb=10_000_000)  # 10 PB will not fit on a laptop


def test_load_aapl_one_day_basic_shape():
    df = load_eq_taq("AAPL", "20200113")
    assert isinstance(df, pl.DataFrame)
    assert len(df) > 1_000_000, f"expected millions of events, got {len(df)}"
    expected_cols = {
        "Date", "Timestamp", "EventType", "Ticker", "Price", "Quantity", "Exchange", "Conditions",
    }
    assert expected_cols.issubset(set(df.columns))


def test_load_aapl_cached_is_fast():
    """After first load builds the parquet cache, subsequent loads must be <5s."""
    # warm cache
    load_eq_taq("AAPL", "20200113")
    t0 = time.perf_counter()
    df = load_eq_taq("AAPL", "20200113")
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"AAPL cached load took {elapsed:.2f}s — Phase 0 success criterion is <5s"
    assert not df.is_empty()


def test_load_eq_taq_missing_ticker_raises():
    with pytest.raises(FileNotFoundError):
        load_eq_taq("NOTAREALTICKER", "20200113")


def test_validate_rejects_empty_dataframe():
    from hft.data.loaders import validate_taq

    empty = pl.DataFrame({"Date": [], "Timestamp": [], "EventType": [], "Ticker": [],
                          "Price": [], "Quantity": [], "Exchange": []})
    with pytest.raises(DataIntegrityError):
        validate_taq(empty, ticker="X", date="20200113")


def test_list_tickers_returns_208():
    """We confirmed earlier that 20200113 has 208 equity tickers."""
    tickers = list_eq_taq_tickers("20200113")
    assert len(tickers) == 208, f"expected 208 tickers, got {len(tickers)}"
    assert "AAPL" in tickers


def test_daily_ohlc_loads():
    df = load_eq_daily_ohlc("20200113")
    assert isinstance(df, pl.DataFrame)
    assert len(df) > 100, "daily OHLC should have hundreds of tickers"
    assert "Ticker" in df.columns
    assert "Close" in df.columns


def test_load_fut_taq_basic_shape():
    df = load_fut_taq("NQH0", "20200113")
    assert isinstance(df, pl.DataFrame)
    assert len(df) > 100_000, "NQ futures should have hundreds of thousands of events"
    assert "Type" in df.columns
    types = set(df["Type"].unique().to_list())
    assert "QUOTE BID" in types
    assert "QUOTE SELL" in types


def test_reconstruct_book_returns_top_levels():
    """Sanity check that reconstruct_book returns the expected structure.

    NOTE: We expect crossed books on this dataset because the feed is event-driven
    without an opening snapshot — see orderbook.py docstring. The function reports
    `is_crossed` honestly; we DO NOT silently filter stale levels here. Phase 1
    EDA will design proper reconstruction (snapshot bootstrapping or TTL).
    """
    df = load_fut_taq("NQH0", "20200113")
    book = reconstruct_book(df, snapshot_time="153000", depth=5)
    assert {"bids", "asks", "is_crossed", "timestamp"}.issubset(book.keys())
    assert len(book["bids"]) <= 5
    assert len(book["asks"]) <= 5
    # Within each side, ordering must be correct (bids descending, asks ascending)
    if len(book["bids"]) >= 2:
        assert book["bids"][0][0] >= book["bids"][1][0], "bids not sorted descending"
    if len(book["asks"]) >= 2:
        assert book["asks"][0][0] <= book["asks"][1][0], "asks not sorted ascending"


def test_nbbo_reconstruction_runs_on_small_slice():
    """Run NBBO on the first 100k AAPL events to verify the algorithm executes."""
    df = load_eq_taq("AAPL", "20200113").head(100_000)
    nbbo = build_nbbo(df, validate_against_published=False)
    assert len(nbbo) > 0
    assert "NBB" in nbbo.columns and "NBO" in nbbo.columns
