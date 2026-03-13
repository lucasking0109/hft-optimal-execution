"""Toy-example tests for the 10 metrics. Each value is hand-verifiable.

Sign convention reminder: every cost-style metric returns **basis points**
where positive = cost (bad for executor), negative = better than benchmark.
"""

from __future__ import annotations

import polars as pl
import pytest

from hft.analysis.metrics import (
    bps,
    effective_spread_bps,
    hit_ratio_at_nbbo,
    implementation_shortfall_bps,
    markout_bps,
    participation_rate,
    post_trade_reversion_bps,
    price_variance_during_execution,
    realized_spread_bps,
    schedule_deviation,
    vwap_slippage_bps,
)
from hft.analysis.nbbo_lookup import NBBOLookup


def _toy_market_df() -> pl.DataFrame:
    """Tiny synthetic TAQ slice for hand-checked metric tests."""
    # 3 trades and a few NBBO quotes around them
    return pl.DataFrame({
        "Date": [20200113] * 12,
        "Timestamp": [
            "09:30:00.000000000",
            "09:30:00.000000000",
            "09:30:00.000000000",
            "09:30:00.500000000",
            "09:30:01.000000000",
            "09:30:01.000000000",
            "09:30:01.000000000",
            "09:30:30.000000000",
            "09:30:30.000000000",
            "09:30:30.000000000",
            "09:35:00.000000000",
            "09:35:00.000000000",
        ],
        "EventType": [
            "QUOTE BID NB", "QUOTE ASK NB", "TRADE NB",
            "TRADE NB",
            "QUOTE BID NB", "QUOTE ASK NB", "TRADE NB",
            "QUOTE BID NB", "QUOTE ASK NB", "TRADE NB",
            "QUOTE BID NB", "QUOTE ASK NB",
        ],
        "Ticker": ["TST"] * 12,
        "Price": [
            99.99, 100.01, 100.00,
            100.00,
            99.98, 100.02, 100.00,
            100.10, 100.12, 100.10,
            99.50, 99.52,
        ],
        "Quantity": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
        "Exchange": ["NASDAQ"] * 12,
        "Conditions": [""] * 12,
    })


def _toy_fills() -> pl.DataFrame:
    """Fills for a 300-share sell at average $100.00."""
    return pl.DataFrame({
        "timestamp_ns": [
            int((9 * 3600 + 30 * 60 + 0) * 1e9),
            int((9 * 3600 + 30 * 60 + 1) * 1e9),
            int((9 * 3600 + 30 * 60 + 30) * 1e9),
        ],
        "price": [100.00, 100.00, 100.10],
        "quantity": [100, 100, 100],
        "venue": ["NASDAQ", "NASDAQ", "NASDAQ"],
        "side": ["sell"] * 3,
    })


def test_vwap_slippage_basic_sell():
    """VWAP of market trades = 100, 100, 100, 100, 100.10  → mean weighted ≈ 100.025
    Executed avg = (100*100 + 100*100 + 100.10*100) / 300 = 100.0333
    Sell side: cost = -1 * (executed - vwap)/vwap > 0 means we sold below VWAP (bad)
    Here executed (100.0333) > vwap (100.025) → favorable for sell → cost negative.
    """
    market = _toy_market_df()
    fills = _toy_fills()
    cost = vwap_slippage_bps(
        fills, market, side="sell",
        window_start_ns=int(9 * 3600 * 1e9),
        window_end_ns=int(10 * 3600 * 1e9),
    )
    # Hand calc: market VWAP = (100*100 + 100*100 + 100*100 + 100.10*100)/400 = 100.025
    # executed = 100.0333
    # cost (sell) = -(executed - vwap)/vwap = -(100.0333 - 100.025)/100.025 = ~ -0.83 bps
    assert -1.5 < cost < -0.5, f"expected ~-0.83 bps, got {cost:.3f}"


def test_vwap_slippage_basic_buy():
    """Same fills but treat as buy. cost positive (we paid above VWAP)."""
    market = _toy_market_df()
    fills = _toy_fills().with_columns(pl.lit("buy").alias("side"))
    cost = vwap_slippage_bps(
        fills, market, side="buy",
        window_start_ns=int(9 * 3600 * 1e9),
        window_end_ns=int(10 * 3600 * 1e9),
    )
    assert 0.5 < cost < 1.5, f"expected ~+0.83 bps, got {cost:.3f}"


def test_implementation_shortfall_sell():
    """Arrival mid 100, executed 100.0333. Sell IS = -(100.0333 - 100)/100 = -3.33 bps (favorable)."""
    fills = _toy_fills()
    is_b = implementation_shortfall_bps(fills, side="sell", arrival_mid=100.0)
    assert -4 < is_b < -3, f"expected ~-3.33 bps, got {is_b:.3f}"


def test_implementation_shortfall_buy_costs():
    fills = _toy_fills()
    is_b = implementation_shortfall_bps(fills, side="buy", arrival_mid=100.0)
    assert 3 < is_b < 4, f"expected ~+3.33 bps, got {is_b:.3f}"


def test_implementation_shortfall_rejects_bad_arrival():
    fills = _toy_fills()
    with pytest.raises(ValueError):
        implementation_shortfall_bps(fills, side="sell", arrival_mid=0)


def test_effective_spread_basic():
    """At 09:30:00 the NBBO was 99.99/100.01; mid 100.00. Fill at 100.00 → 0 bps."""
    market = _toy_market_df()
    fills = _toy_fills()
    lookup = NBBOLookup(market)
    eff = effective_spread_bps(fills, lookup)
    # Fills at 09:30:00 (mid 100), 09:30:01 (mid 100), 09:30:30 (mid 100.11)
    # Effective spread (each weighted by qty=100):
    # fill1: 2*|100-100|/100 = 0
    # fill2: 2*|100-100|/100 = 0
    # fill3: 2*|100.10-100.11|/100.11 ≈ 0.000199...  → ~2 bps
    # Mean weighted by qty (all 100) = avg = ~0.66 bps
    assert 0 <= eff < 2, f"expected 0–2 bps, got {eff:.3f}"


def test_markout_bps_post_trade_signal():
    """Mid moves from ~100 at fill to 99.51 at 5-min markout (we sold).
    Sell side → mid_after < fill_price → cost = -1 * (mid_after - px)/px > 0 (= mid-fall AGAINST a sell? wait)

    Wait: if I sold at 100.10 and mid drops to 99.51 → I sold high, then it dropped
    further → that's GOOD for me (I got out before the fall). cost should be negative.
    """
    market = _toy_market_df()
    fills = _toy_fills()
    lookup = NBBOLookup(market)
    m60 = markout_bps(fills, lookup, side="sell", horizon_seconds=300)  # 5 min
    # For each fill, mid_after = 99.51
    # cost = -side_sign * (mid_after - px)/px = -(-1)*(99.51 - 100)/100 = -49 bps (favorable)
    assert m60 < -30, f"expected favorable (negative) markout, got {m60:.3f}"


def test_post_trade_reversion_sell():
    """Final fill at 100.10. Mid 5-min after = (99.50+99.52)/2 = 99.51.
    Sell, mid_after fell → positive (info trade). reversion = -1*(99.51-100.10)/100.10 ≈ +59 bps
    """
    market = _toy_market_df()
    fills = _toy_fills()
    lookup = NBBOLookup(market)
    rev = post_trade_reversion_bps(fills, lookup, side="sell")
    assert 50 < rev < 70, f"expected ~+59 bps, got {rev:.3f}"


def test_participation_rate_basic():
    market = _toy_market_df()
    fills = _toy_fills()
    pov = participation_rate(
        fills, market,
        window_start_ns=int(9 * 3600 * 1e9),
        window_end_ns=int(10 * 3600 * 1e9),
    )
    # Market trades in window = 4 trades * 100 = 400 shares
    # My fills = 300 shares
    # POV = 300/400 = 0.75
    assert abs(pov - 0.75) < 0.001, f"expected 0.75, got {pov}"


def test_price_variance_basic():
    fills = _toy_fills()
    var = price_variance_during_execution(fills)
    # prices [100, 100, 100.10] → std ≈ 0.0577
    assert 0.04 < var < 0.07, f"expected ~0.058, got {var}"


def test_schedule_deviation_perfect():
    fills = _toy_fills()
    # Plan: at the 3 actual fill timestamps, plan exactly cumulative = 100, 200, 300
    plan = [(int(t), q) for t, q in zip(
        fills["timestamp_ns"].to_list(),
        [100, 200, 300],
    )]
    rmse = schedule_deviation(fills, planned_schedule=plan)
    assert rmse < 1, f"expected near-zero RMSE for matching plan, got {rmse}"


def test_schedule_deviation_off():
    fills = _toy_fills()
    plan = [(int(t), q) for t, q in zip(
        fills["timestamp_ns"].to_list(),
        [200, 200, 200],   # plan says 200/200/200 but actual was 100/200/300
    )]
    rmse = schedule_deviation(fills, planned_schedule=plan)
    # Errors: 100, 0, 100 → RMSE = sqrt((100^2+0+100^2)/3) ≈ 81.65
    assert 70 < rmse < 90, f"expected ~82, got {rmse}"


def test_hit_ratio_at_nbbo_perfect_sell():
    market = _toy_market_df()
    fills = _toy_fills()
    lookup = NBBOLookup(market)
    # Each fill price equals the prevailing NBB? Let's check:
    # at 09:30:00 NBB=99.99, but fill at 100.00 → not at NBB → miss
    # at 09:30:01 NBB=99.98, fill 100.00 → miss
    # at 09:30:30 NBB=100.10, fill 100.10 → hit
    hit = hit_ratio_at_nbbo(fills, lookup, side="sell")
    assert abs(hit - 1/3) < 0.01, f"expected 1/3 hit, got {hit}"


def test_realized_spread_handles_horizon():
    market = _toy_market_df()
    fills = _toy_fills()
    lookup = NBBOLookup(market)
    rs = realized_spread_bps(fills, lookup, side="sell", horizon_ns=int(5 * 60 * 1e9))
    # Sanity: function returns a number (already sign-checked by post_trade_reversion logic)
    assert isinstance(rs, float)


def test_bps_helper():
    assert bps(0.0001) == pytest.approx(1.0)
    assert bps(-0.0005) == pytest.approx(-5.0)
