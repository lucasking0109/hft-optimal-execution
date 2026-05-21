"""Phase F — regression guard against same-day volume_profile leakage.

These tests enforce that:
1. BacktestEngine.market_context() warns loudly if no override is provided.
2. Passing an override silences the warning.
3. compute_lookback_volume_profile() refuses to load the eval day's parquet
   (excluded by caller).
4. The lookback function actually returns averaged-across-days numbers.
"""

from __future__ import annotations

import warnings

import polars as pl
import pytest

from hft.analysis.vwap import compute_lookback_volume_profile
from hft.backtest.engine import BacktestEngine


def test_market_context_warns_without_override():
    """Default same-day behavior must emit UserWarning (look-ahead bias)."""
    engine = BacktestEngine("AAPL", "20200113")
    with pytest.warns(UserWarning, match="look-ahead bias"):
        engine.market_context()


def test_market_context_silent_with_override():
    """Passing an explicit override should be silent (no warning)."""
    engine = BacktestEngine("AAPL", "20200113")
    fake_profile = pl.DataFrame({
        "bucket_min": [114, 115, 116],
        "volume": [1000.0, 800.0, 1200.0],
    })
    with warnings.catch_warnings():
        warnings.simplefilter("error")   # any warning → fail
        ctx = engine.market_context(volume_profile_override=fake_profile)
    assert ctx["volume_profile"].equals(fake_profile)


def test_lookback_profile_requires_nonempty_dates():
    """Empty lookback_dates must raise loud ValueError (NO silent fallback)."""
    with pytest.raises(ValueError, match="lookback_dates is empty"):
        compute_lookback_volume_profile("AAPL", lookback_dates=[])


def test_lookback_profile_excludes_eval_date(tmp_path):
    """If we ask for lookback=[D1, D2, D3] the function must NOT load
    a separate eval_date (D4) — we test by giving it a fake cache root
    that only has D1/D2/D3 files, no D4. Function should succeed.
    """
    # Build a fake cache root with 2 days; 'eval day' isn't in it.
    for d in ["20200113", "20200114"]:
        day_dir = tmp_path / d
        day_dir.mkdir()
        df = pl.DataFrame({
            "Timestamp": pl.Series(["09:35:00.000000000"] * 3),
            "EventType": ["TRADE"] * 3,
            "Quantity": [100, 200, 300],
            "Price": [100.0, 100.5, 101.0],
        })
        df.write_parquet(day_dir / "AAPL.parquet")
    # Should succeed using only D1 and D2 — does NOT need a 'D5' file.
    prof = compute_lookback_volume_profile(
        "AAPL", lookback_dates=["20200113", "20200114"], cache_root=tmp_path,
    )
    assert prof.height >= 1
    assert "volume" in prof.columns
    assert "bucket_min" in prof.columns


def test_lookback_profile_actually_averages(tmp_path):
    """Build 2 fake days with known volumes; result must be mean per bucket."""
    # Day 1: bucket 114 has 1000 shares
    d1 = tmp_path / "20200113"
    d1.mkdir()
    pl.DataFrame({
        "Timestamp": ["09:30:30.000000000"] * 2,
        "EventType": ["TRADE", "TRADE"],
        "Quantity": [400, 600],
        "Price": [100.0, 100.0],
    }).write_parquet(d1 / "AAPL.parquet")
    # Day 2: bucket 114 has 2000 shares
    d2 = tmp_path / "20200114"
    d2.mkdir()
    pl.DataFrame({
        "Timestamp": ["09:30:30.000000000"] * 2,
        "EventType": ["TRADE", "TRADE"],
        "Quantity": [1000, 1000],
        "Price": [100.0, 100.0],
    }).write_parquet(d2 / "AAPL.parquet")
    prof = compute_lookback_volume_profile(
        "AAPL", lookback_dates=["20200113", "20200114"], cache_root=tmp_path,
    )
    # Mean of 1000 and 2000 = 1500
    bucket_114_vol = prof.filter(pl.col("bucket_min") == 114)["volume"][0]
    assert abs(bucket_114_vol - 1500) < 1e-6, (
        f"expected mean 1500, got {bucket_114_vol}"
    )


def test_lookback_profile_raises_on_all_missing(tmp_path):
    """If none of the lookback dates have a parquet, raise loudly."""
    with pytest.raises(RuntimeError, match="no parquet loadable"):
        compute_lookback_volume_profile(
            "AAPL", lookback_dates=["19990101"], cache_root=tmp_path,
        )
