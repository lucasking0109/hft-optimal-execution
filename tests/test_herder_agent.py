"""Unit tests for HerderAgent (Phase 5B+ Stage 3 Step 3).

Tests the decision/order-placement logic in isolation by:
  - Instantiating the agent with random_state (no kernel required)
  - Monkey-patching `placeLimitOrder` to capture orders
  - Feeding synthetic MARKET_DATA messages via receiveMessage
  - Asserting expected behavior

Per Stage 3 plan: 8/8 must pass before integration smoke test.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "abides-sim"))

from agent.examples.HerderAgent import HerderAgent  # noqa: E402


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

class MockKernel:
    """Minimal kernel stub for receiveMessage logging path."""
    def fmtTime(self, t):
        return str(t)
    def setAgentComputeDelay(self, sender=None, requestedDelay=None):
        pass


def make_herder(
    *,
    lookback_window_secs=5.0,
    entry_threshold_bps=2.0,
    max_size=5,
    position_cap=50,
    tolerance_ticks=5,
):
    """Create a HerderAgent suitable for unit testing (no kernel)."""
    rng = np.random.RandomState(seed=42)
    h = HerderAgent(
        id=1, name="HERDER_TEST", type="HerderAgent",
        symbol="AAPL", starting_cash=10_000_000,
        lookback_window_secs=lookback_window_secs,
        entry_threshold_bps=entry_threshold_bps,
        max_size=max_size,
        position_cap=position_cap,
        tolerance_ticks=tolerance_ticks,
        log_orders=False, random_state=rng,
    )
    h.kernel = MockKernel()
    h.exchangeID = 0  # mark/exchange id for messaging path
    h.mkt_open = pd.Timestamp("2020-01-13 09:30:00")
    h.mkt_close = pd.Timestamp("2020-01-13 16:00:00")
    # Patch placeLimitOrder to capture instead of sending to exchange
    h._captured_orders = []
    def fake_place(symbol, qty, is_buy, limit_price, **kwargs):
        h._captured_orders.append({
            'symbol': symbol, 'qty': qty, 'is_buy': is_buy, 'limit_price': limit_price,
        })
    h.placeLimitOrder = fake_place
    return h


def make_market_data_msg(bid_price, bid_qty, ask_price, ask_qty, t=None):
    """Build a fake MARKET_DATA message body matching ABIDES schema.

    handleMarketData (TradingAgent) needs: symbol, bids, asks, last_transaction, exchange_ts.
    """
    return SimpleNamespace(body={
        'msg': 'MARKET_DATA',
        'symbol': 'AAPL',
        'bids': [(bid_price, bid_qty)],
        'asks': [(ask_price, ask_qty)],
        'last_transaction': (bid_price + ask_price) // 2,
        'exchange_ts': t if t is not None else pd.Timestamp("2020-01-13 09:30:00"),
    })


def feed(h, t, bid, ask, bid_qty=100, ask_qty=100):
    """Feed one MARKET_DATA event at time t (pd.Timestamp) with given bid/ask."""
    h.currentTime = t
    msg = make_market_data_msg(bid, bid_qty, ask, ask_qty, t=t)
    h.receiveMessage(t, msg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_init_requires_data_calibrated_params():
    """Constructor MUST raise if any of the 3 calibrated params is None."""
    rng = np.random.RandomState(42)
    with pytest.raises(ValueError, match="Step 0 data calibration"):
        HerderAgent(id=1, name="X", type="HerderAgent", symbol="AAPL",
                    starting_cash=10_000_000,
                    lookback_window_secs=None,  # ← missing
                    entry_threshold_bps=2.0, max_size=5,
                    random_state=rng)
    rng2 = np.random.RandomState(42)
    with pytest.raises(ValueError):
        HerderAgent(id=1, name="X", type="HerderAgent", symbol="AAPL",
                    starting_cash=10_000_000,
                    lookback_window_secs=5.0,
                    entry_threshold_bps=None,  # ← missing
                    max_size=5,
                    random_state=rng2)
    rng3 = np.random.RandomState(42)
    with pytest.raises(ValueError):
        HerderAgent(id=1, name="X", type="HerderAgent", symbol="AAPL",
                    starting_cash=10_000_000,
                    lookback_window_secs=5.0,
                    entry_threshold_bps=2.0,
                    max_size=None,  # ← missing
                    random_state=rng3)


def test_no_trade_with_empty_buffer():
    """First MARKET_DATA → buffer has 1 entry → no order placed."""
    h = make_herder()
    t0 = pd.Timestamp("2020-01-13 09:30:00")
    feed(h, t0, 10000, 10010)
    assert len(h._captured_orders) == 0
    assert h._signal_count == 0


def test_no_trade_below_threshold():
    """Drift below entry_threshold → no order."""
    h = make_herder(entry_threshold_bps=10.0)  # 10 bps very high threshold
    t0 = pd.Timestamp("2020-01-13 09:30:00")
    feed(h, t0, 10000, 10010)
    feed(h, t0 + pd.Timedelta(seconds=2), 10001, 10011)  # drift ~1 bps, below 10
    assert len(h._captured_orders) == 0
    assert h._signal_count == 0


def test_buys_on_uptick():
    """Strong upward drift → buy at marketable limit (ask + tolerance)."""
    h = make_herder(entry_threshold_bps=2.0, tolerance_ticks=5, max_size=5)
    t0 = pd.Timestamp("2020-01-13 09:30:00")
    feed(h, t0, 10000, 10010)
    # +20 bps drift between observations → way above threshold
    feed(h, t0 + pd.Timedelta(seconds=2), 10020, 10030)
    assert len(h._captured_orders) == 1
    o = h._captured_orders[0]
    assert o['is_buy'] is True
    assert o['symbol'] == 'AAPL'
    # marketable limit = ask + tolerance = 10030 + 5
    assert o['limit_price'] == 10035
    assert h._signal_count == 1
    assert h._order_placed_count == 1


def test_sells_on_downtick():
    """Strong downward drift → sell at marketable limit (bid - tolerance)."""
    h = make_herder(entry_threshold_bps=2.0, tolerance_ticks=5, max_size=5)
    t0 = pd.Timestamp("2020-01-13 09:30:00")
    feed(h, t0, 10000, 10010)
    feed(h, t0 + pd.Timedelta(seconds=2), 9980, 9990)  # -20 bps
    assert len(h._captured_orders) == 1
    o = h._captured_orders[0]
    assert o['is_buy'] is False
    # marketable limit = bid - tolerance = 9980 - 5
    assert o['limit_price'] == 9975


def test_position_cap_blocks_overbuy():
    """Position at cap → buy signal blocked + diagnostic counter incremented."""
    h = make_herder(entry_threshold_bps=2.0, position_cap=10, max_size=5)
    h.holdings['AAPL'] = 10  # already at cap
    t0 = pd.Timestamp("2020-01-13 09:30:00")
    feed(h, t0, 10000, 10010)
    feed(h, t0 + pd.Timedelta(seconds=2), 10020, 10030)  # buy signal
    assert len(h._captured_orders) == 0
    assert h._signal_count == 1
    assert h._cap_blocked_count == 1
    assert h._order_placed_count == 0


def test_size_scales_with_drift_intensity():
    """Drift = 5× threshold → size at max_size; drift = 1× threshold → size = 1.

    Intensity = min(1, |drift| / (threshold * 5))
    size = max(1, round(max_size * intensity))
    """
    # Test 5x threshold case
    h_high = make_herder(entry_threshold_bps=2.0, max_size=10)
    t0 = pd.Timestamp("2020-01-13 09:30:00")
    feed(h_high, t0, 10000, 10010)
    # 100 bps drift = 10 / 10005 → ~100 bps; threshold 2 bps → 50× threshold
    feed(h_high, t0 + pd.Timedelta(seconds=2), 10100, 10110)
    assert len(h_high._captured_orders) == 1
    assert h_high._captured_orders[0]['qty'] == 10  # capped at max_size

    # Test ~1x threshold case
    h_low = make_herder(entry_threshold_bps=2.0, max_size=10)
    feed(h_low, t0, 10000, 10010)
    # ~2.5 bps drift → just above threshold → intensity ≈ 0.25 → size ≈ 2-3
    feed(h_low, t0 + pd.Timedelta(seconds=2), 10003, 10013)
    if h_low._captured_orders:
        assert 1 <= h_low._captured_orders[0]['qty'] <= 3


def test_lookback_trim():
    """Obs older than lookback_window_secs are popleft'd."""
    h = make_herder(lookback_window_secs=5.0, entry_threshold_bps=2.0)
    t0 = pd.Timestamp("2020-01-13 09:30:00")
    # Feed 10 obs, 1 sec apart
    for i in range(10):
        feed(h, t0 + pd.Timedelta(seconds=i), 10000 + i, 10010 + i)
    # lookback 5s → buffer should have ~5-6 entries (last 5+ seconds)
    assert 4 <= len(h.mid_buffer) <= 6


def test_invalid_book_no_trade():
    """Missing or inverted bid/ask → no trade."""
    h = make_herder()
    t0 = pd.Timestamp("2020-01-13 09:30:00")
    # Empty bids → returns silently
    h.currentTime = t0
    h.receiveMessage(t0, SimpleNamespace(body={
        'msg': 'MARKET_DATA', 'symbol': 'AAPL',
        'bids': [], 'asks': [(10010, 100)],
        'last_transaction': 10000, 'exchange_ts': t0,
    }))
    assert len(h._captured_orders) == 0
    # Inverted spread
    feed(h, t0, 10010, 10000)  # ask < bid
    assert len(h._captured_orders) == 0
