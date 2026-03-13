"""10 execution-quality metrics (industry standard).

Sign convention: each cost-style metric returns **basis points**, where
**positive = cost (bad for the executor)**, negative = better than benchmark.
This works for both buy and sell once we multiply by side_sign internally.

side: +1 for buy, -1 for sell.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

from hft.analysis.nbbo_lookup import NBBOLookup
from hft.analysis.vwap import compute_market_vwap

Side = Literal["buy", "sell"]


def _side_sign(side: Side) -> int:
    return 1 if side == "buy" else -1


def bps(x: float) -> float:
    """Convert a fractional return to basis points."""
    return x * 10000


# ---------------------------------------------------------------------------
# 1. VWAP slippage
# ---------------------------------------------------------------------------

def vwap_slippage_bps(
    fills: pl.DataFrame,
    market_df: pl.DataFrame,
    *,
    side: Side,
    window_start_ns: int,
    window_end_ns: int,
) -> float:
    """Cost vs market VWAP over the same execution window.

    Positive = cost (you executed worse than market average).
    """
    if fills.is_empty():
        raise ValueError("Empty fills — cannot compute VWAP slippage")
    executed_avg = float(
        (fills["price"] * fills["quantity"]).sum() / fills["quantity"].sum()
    )
    market_vwap = compute_market_vwap(
        market_df, start_ns=window_start_ns, end_ns=window_end_ns
    )
    return bps(_side_sign(side) * (executed_avg - market_vwap) / market_vwap)


# ---------------------------------------------------------------------------
# 2. Implementation Shortfall (IS)
# ---------------------------------------------------------------------------

def implementation_shortfall_bps(
    fills: pl.DataFrame,
    *,
    side: Side,
    arrival_mid: float,
) -> float:
    """Cost vs the mid-price at the moment the parent order was decided.

    Positive = cost.
    """
    if fills.is_empty():
        raise ValueError("Empty fills — cannot compute IS")
    if arrival_mid <= 0:
        raise ValueError(f"arrival_mid must be > 0, got {arrival_mid}")
    executed_avg = float(
        (fills["price"] * fills["quantity"]).sum() / fills["quantity"].sum()
    )
    return bps(_side_sign(side) * (executed_avg - arrival_mid) / arrival_mid)


# ---------------------------------------------------------------------------
# 3. Effective spread
# ---------------------------------------------------------------------------

def effective_spread_bps(fills: pl.DataFrame, lookup: NBBOLookup) -> float:
    """Mean 2 * |fill_price − mid_at_fill| / mid, weighted by quantity."""
    if fills.is_empty():
        raise ValueError("Empty fills")
    weighted_sum = 0.0
    total_qty = 0
    for ns, px, qty in zip(
        fills["timestamp_ns"].to_list(), fills["price"].to_list(), fills["quantity"].to_list()
    ):
        mid = lookup.mid_at(ns)
        if mid is None:
            continue
        spread_frac = 2 * abs(px - mid) / mid
        weighted_sum += spread_frac * qty
        total_qty += qty
    if total_qty == 0:
        raise ValueError("No fills with valid mid for effective spread")
    return bps(weighted_sum / total_qty)


# ---------------------------------------------------------------------------
# 4. Realized spread (5-minute markout-adjusted)
# ---------------------------------------------------------------------------

def realized_spread_bps(
    fills: pl.DataFrame,
    lookup: NBBOLookup,
    *,
    side: Side,
    horizon_ns: int = 5 * 60 * 1_000_000_000,
) -> float:
    """2 × side × (fill_price − mid_after_horizon) / mid, weighted by quantity.

    Negative = the price moved further your way after fill (free money,
    you were aggressive but right).
    """
    if fills.is_empty():
        raise ValueError("Empty fills")
    s = _side_sign(side)
    weighted_sum = 0.0
    total_qty = 0
    for ns, px, qty in zip(
        fills["timestamp_ns"].to_list(), fills["price"].to_list(), fills["quantity"].to_list()
    ):
        mid_after = lookup.mid_at(ns + horizon_ns)
        if mid_after is None or mid_after <= 0:
            continue
        # Sign chosen so positive = cost
        rs = s * (px - mid_after) / mid_after
        weighted_sum += rs * qty * 2  # ×2 for round-trip convention
        total_qty += qty
    if total_qty == 0:
        raise ValueError("No fills with valid post-horizon mid")
    return bps(weighted_sum / total_qty)


# ---------------------------------------------------------------------------
# 5. Markout (1s / 10s / 60s)
# ---------------------------------------------------------------------------

def markout_bps(
    fills: pl.DataFrame,
    lookup: NBBOLookup,
    *,
    side: Side,
    horizon_seconds: int,
) -> float:
    """Cost-positive markout: average (mid_after − fill_price) / fill_price × side.

    Positive = price moved adversely after your fill (information leakage).
    Negative = price moved your way (free profit).
    """
    if fills.is_empty():
        raise ValueError("Empty fills")
    s = _side_sign(side)
    horizon_ns = horizon_seconds * 1_000_000_000
    weighted_sum = 0.0
    total_qty = 0
    for ns, px, qty in zip(
        fills["timestamp_ns"].to_list(), fills["price"].to_list(), fills["quantity"].to_list()
    ):
        mid_after = lookup.mid_at(ns + horizon_ns)
        if mid_after is None or mid_after <= 0:
            continue
        # For sell (s=-1): if price drops after we sold (mid_after < px), we lost = cost
        # cost_frac = -1 * (mid_after - px) / px  → positive when mid_after < px (price fell after sell)
        # General: cost = -s * (mid_after - px) / px
        weighted_sum += -s * (mid_after - px) / px * qty
        total_qty += qty
    if total_qty == 0:
        raise ValueError("No fills with valid post-horizon mid")
    return bps(weighted_sum / total_qty)


# ---------------------------------------------------------------------------
# 6. Post-trade reversion (5-min after final fill)
# ---------------------------------------------------------------------------

def post_trade_reversion_bps(
    fills: pl.DataFrame,
    lookup: NBBOLookup,
    *,
    side: Side,
    horizon_ns: int = 5 * 60 * 1_000_000_000,
) -> float:
    """After all child orders done, did mid revert toward your starting price?

    Positive = mid kept moving away → your trades had real information.
    Negative = mid reverted → you over-paid for liquidity.
    """
    if fills.is_empty():
        raise ValueError("Empty fills")
    final_ns = int(fills["timestamp_ns"].max())
    final_px = float(fills["price"].tail(1).item())
    mid_after = lookup.mid_at(final_ns + horizon_ns)
    if mid_after is None:
        raise ValueError("No mid available 5 min after final fill")
    s = _side_sign(side)
    # If we bought (s=+1) and mid went UP after we finished → positive (info)
    # If we sold (s=-1) and mid went DOWN after we finished → positive
    return bps(s * (mid_after - final_px) / final_px)


# ---------------------------------------------------------------------------
# 7. Participation Rate (POV)
# ---------------------------------------------------------------------------

def participation_rate(
    fills: pl.DataFrame,
    market_df: pl.DataFrame,
    *,
    window_start_ns: int,
    window_end_ns: int,
) -> float:
    """Sum of my fill quantity / sum of total market trade volume in window."""
    if fills.is_empty():
        return 0.0
    my_qty = int(fills["quantity"].sum())
    if "ns_of_day" not in market_df.columns:
        from hft.data.timeparse import add_eq_ns_of_day
        market_df = add_eq_ns_of_day(market_df)
    mkt_trades = market_df.filter(
        pl.col("EventType").is_in(["TRADE", "TRADE NB"])
        & (pl.col("ns_of_day") >= window_start_ns)
        & (pl.col("ns_of_day") <= window_end_ns)
    )
    market_qty = int(mkt_trades["Quantity"].sum())
    if market_qty <= 0:
        raise ValueError("Market volume is zero in execution window — cannot compute POV.")
    return my_qty / market_qty


# ---------------------------------------------------------------------------
# 8. Price variance during execution
# ---------------------------------------------------------------------------

def price_variance_during_execution(fills: pl.DataFrame) -> float:
    """Standard deviation of fill prices (in dollars). Higher = more risk taken."""
    if fills.is_empty():
        raise ValueError("Empty fills")
    if len(fills) < 2:
        return 0.0
    return float(fills["price"].std())


# ---------------------------------------------------------------------------
# 9. Schedule deviation
# ---------------------------------------------------------------------------

def schedule_deviation(
    fills: pl.DataFrame,
    *,
    planned_schedule: list[tuple[int, int]],  # list of (ns_of_day, planned_cumulative_qty)
) -> float:
    """RMSE between actual cumulative quantity and planned cumulative quantity.

    Useful for measuring how well the engine stuck to the strategy's plan.
    """
    if fills.is_empty():
        raise ValueError("Empty fills")
    if not planned_schedule:
        raise ValueError("Planned schedule is empty")

    # Build actual cumulative quantity at each planned timestamp
    fills_sorted = fills.sort("timestamp_ns")
    fill_ns = fills_sorted["timestamp_ns"].to_list()
    fill_qty = fills_sorted["quantity"].to_list()

    sq_err = 0.0
    n = 0
    cum = 0
    fill_idx = 0
    for plan_ns, plan_cum in planned_schedule:
        while fill_idx < len(fill_ns) and fill_ns[fill_idx] <= plan_ns:
            cum += fill_qty[fill_idx]
            fill_idx += 1
        sq_err += (cum - plan_cum) ** 2
        n += 1
    return (sq_err / n) ** 0.5 if n > 0 else 0.0


# ---------------------------------------------------------------------------
# 10. Hit ratio at NBBO
# ---------------------------------------------------------------------------

def hit_ratio_at_nbbo(
    fills: pl.DataFrame,
    lookup: NBBOLookup,
    *,
    side: Side,
    tolerance: float = 1e-4,
) -> float:
    """Fraction of fills whose price equals the NBBO best at fill time."""
    if fills.is_empty():
        raise ValueError("Empty fills")
    hits = 0
    for ns, px in zip(fills["timestamp_ns"].to_list(), fills["price"].to_list()):
        snap = lookup.at(ns)
        target = snap.nbb if side == "sell" else snap.nbo
        if target is None:
            continue
        if abs(px - target) <= max(tolerance, target * 1e-6):
            hits += 1
    return hits / len(fills)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

@dataclass
class MetricsResult:
    vwap_slip_bps: float
    is_bps: float
    effective_spread_bps: float
    realized_spread_bps: float
    markout_1s_bps: float
    markout_10s_bps: float
    markout_60s_bps: float
    reversion_5m_bps: float
    pov: float
    price_var: float
    schedule_rmse: float
    hit_ratio_nbbo: float

    def to_row(self, **labels) -> dict:
        d = self.__dict__.copy()
        d.update(labels)
        return d


def compute_all_metrics(
    fills: pl.DataFrame,
    market_df: pl.DataFrame,
    lookup: NBBOLookup,
    *,
    side: Side,
    arrival_mid: float,
    window_start_ns: int,
    window_end_ns: int,
    planned_schedule: list[tuple[int, int]] | None = None,
) -> MetricsResult:
    """Compute all 10 metrics. Each metric raises explicitly on failure
    (NO Silent Fallback)."""
    try:
        vwap_slip = vwap_slippage_bps(fills, market_df, side=side,
                                      window_start_ns=window_start_ns,
                                      window_end_ns=window_end_ns)
    except Exception as e:
        raise ValueError(f"vwap_slippage_bps failed: {e}") from e

    is_b = implementation_shortfall_bps(fills, side=side, arrival_mid=arrival_mid)
    eff = effective_spread_bps(fills, lookup)
    rls = realized_spread_bps(fills, lookup, side=side)
    m1 = markout_bps(fills, lookup, side=side, horizon_seconds=1)
    m10 = markout_bps(fills, lookup, side=side, horizon_seconds=10)
    m60 = markout_bps(fills, lookup, side=side, horizon_seconds=60)
    rev = post_trade_reversion_bps(fills, lookup, side=side)
    pov = participation_rate(fills, market_df,
                             window_start_ns=window_start_ns,
                             window_end_ns=window_end_ns)
    var = price_variance_during_execution(fills)
    rmse = schedule_deviation(fills, planned_schedule=planned_schedule) if planned_schedule else 0.0
    hit = hit_ratio_at_nbbo(fills, lookup, side=side)

    return MetricsResult(
        vwap_slip_bps=vwap_slip,
        is_bps=is_b,
        effective_spread_bps=eff,
        realized_spread_bps=rls,
        markout_1s_bps=m1,
        markout_10s_bps=m10,
        markout_60s_bps=m60,
        reversion_5m_bps=rev,
        pov=pov,
        price_var=var,
        schedule_rmse=rmse,
        hit_ratio_nbbo=hit,
    )
