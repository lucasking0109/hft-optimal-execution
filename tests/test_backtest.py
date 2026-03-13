"""End-to-end backtest engine tests.

These exercise the full pipeline: load AAPL → create parent order → run
TWAP → produce metrics. Real-data integration test.
"""

from __future__ import annotations

import pytest

from hft.backtest.engine import BacktestEngine
from hft.strategies.base import ParentOrder
from hft.strategies.twap import TWAPStrategy
from hft.strategies.vwap_following import VWAPFollowingStrategy

NS_PER_HOUR = 3600 * 1_000_000_000


def _parent(qty: int = 50_000) -> ParentOrder:
    return ParentOrder(
        ticker="AAPL", date="20200113", side="sell",
        quantity=qty,
        start_ns=int(10 * NS_PER_HOUR),     # 10:00
        end_ns=int(11 * NS_PER_HOUR),       # 11:00
    )


@pytest.fixture(scope="module")
def engine():
    return BacktestEngine("AAPL", "20200113")


def test_twap_backtest_completes(engine):
    parent = _parent(qty=10_000)
    strat = TWAPStrategy(num_slices=20)
    res = engine.run(parent, strat)
    assert res.fills.shape[0] > 0
    assert int(res.fills["quantity"].sum()) == 10_000
    # Fills should all be in the window
    assert int(res.fills["timestamp_ns"].min()) >= parent.start_ns
    assert int(res.fills["timestamp_ns"].max()) <= parent.end_ns


def test_twap_backtest_metrics_have_sane_magnitude(engine):
    parent = _parent(qty=10_000)
    strat = TWAPStrategy(num_slices=20)
    res = engine.run(parent, strat)
    m = res.metrics
    # Sanity ranges (cost/bps shouldn't be insane)
    assert -1000 < m.vwap_slip_bps < 1000
    assert -1000 < m.is_bps < 1000
    assert m.effective_spread_bps >= 0
    assert 0 <= m.pov <= 1
    assert 0 <= m.hit_ratio_nbbo <= 1


def test_vwap_following_backtest_completes(engine):
    parent = _parent(qty=10_000)
    strat = VWAPFollowingStrategy()
    ctx = engine.market_context(bin_minutes=5)
    res = engine.run(parent, strat, market_context=ctx)
    assert int(res.fills["quantity"].sum()) == 10_000
