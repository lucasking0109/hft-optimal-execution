"""Backtest engine — replay child orders against tick data NBBO.

KNOWN SIMPLIFICATION (documented per NO Silent Fallback principle):
- Each child order is treated as a marketable order that fills entirely at the
  prevailing NBBO best price (sell → NBB; buy → NBO).
- We do NOT model self-impact (your child order pushing the price), partial
  fills, queue position, or latency.
- Quantities larger than the prevailing best-side size are filled anyway at
  best price (`is_oversize` flag is recorded so post-hoc analysis can detect).

These are explicit assumptions that bias slippage downward (more optimistic
than reality). They will be revisited in Phase 5 ABIDES integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import polars as pl

from hft.analysis.metrics import MetricsResult, compute_all_metrics
from hft.analysis.nbbo_lookup import NBBOLookup
from hft.analysis.vwap import compute_market_vwap, compute_volume_profile
from hft.data import load_eq_taq
from hft.data.timeparse import add_eq_ns_of_day
from hft.strategies.base import ExecutionStrategy, Fill, ParentOrder


@dataclass
class BacktestResult:
    parent: ParentOrder
    strategy_name: str
    fills: pl.DataFrame
    metrics: MetricsResult
    arrival_mid: float
    market_vwap: float
    oversize_count: int  # number of child orders larger than best-side size
    null_nbbo_count: int  # children where NBBO unavailable (skipped)


class BacktestEngine:
    """Per (ticker, date) backtester. Build once, run many strategies."""

    def __init__(self, ticker: str, date: str, *, market_df: pl.DataFrame | None = None):
        self.ticker = ticker
        self.date = date
        self.df = market_df if market_df is not None else load_eq_taq(ticker, date)
        if "ns_of_day" not in self.df.columns:
            self.df = add_eq_ns_of_day(self.df)
        self.lookup = NBBOLookup(self.df)

    def market_context(
        self, *,
        bin_minutes: int = 5,
        volume_profile_override: pl.DataFrame | None = None,
    ) -> dict:
        """Build market context dict consumed by volume-aware strategies.

        Args:
            bin_minutes: bucket granularity for volume profile.
            volume_profile_override: pre-computed (causal, no look-ahead)
                DataFrame[bucket_min, volume] — typically from
                `compute_lookback_volume_profile()`. Recommended for
                OOS evaluation. If None, falls back to same-day volume
                profile and emits a UserWarning (look-ahead bias; kept
                for backward compatibility with older test code).
        """
        if volume_profile_override is not None:
            profile = volume_profile_override
        else:
            import warnings
            warnings.warn(
                "BacktestEngine.market_context() called without "
                "volume_profile_override — using same-day volume_profile "
                "(look-ahead bias). Pass `volume_profile_override` from "
                "compute_lookback_volume_profile() for OOS-clean results.",
                UserWarning, stacklevel=2,
            )
            profile = compute_volume_profile(
                self.df, bin_minutes=bin_minutes, rth_only=True,
            )
        return {
            "volume_profile": profile,
            "bin_minutes": bin_minutes,
        }

    def run(
        self,
        parent: ParentOrder,
        strategy: ExecutionStrategy,
        *,
        market_context: dict | None = None,
        venue_aware_fills: bool = False,
    ) -> BacktestResult:
        """Run a strategy.

        venue_aware_fills (Phase C):
          - False (default): fill at NBBO best price (legacy behavior).
          - True: if a child has `target_venue` set, fill at that venue's
            local BBO instead. Falls back to NBBO if venue has no quote
            at the timestamp (counted in `venue_fallback`).
        """
        ctx = market_context if market_context is not None else self.market_context()
        children = strategy.schedule(parent, market_context=ctx)
        if not children:
            raise ValueError(
                f"Strategy {strategy.name} returned no child orders. "
                f"This is a strategy bug — investigate before retrying."
            )

        # Arrival mid at start_ns
        arrival = self.lookup.mid_at(parent.start_ns)
        if arrival is None or arrival <= 0:
            raise ValueError(
                f"NBBO mid unavailable at start_ns={parent.start_ns}. "
                f"Window may be outside the data's coverage."
            )

        # Build per-venue lookup lazily (only if venue-aware fills requested)
        venue_lookup = None
        if venue_aware_fills:
            from hft.analysis.nbbo_lookup import VenueBBOLookup
            venue_lookup = VenueBBOLookup(self.df)

        # Simulate fills
        oversize = 0
        null_nbbo = 0
        venue_fallback = 0
        fill_rows = []
        for c in children:
            target_venue = getattr(c, "target_venue", None)
            price = None
            size_at_best = None
            venue = None

            if venue_aware_fills and target_venue and venue_lookup is not None:
                v_bbo = venue_lookup.at_venue(c.timestamp_ns, target_venue)
                if parent.side == "sell" and v_bbo.bid is not None and v_bbo.bid > 0:
                    price, size_at_best, venue = v_bbo.bid, v_bbo.bid_size, target_venue
                elif parent.side == "buy" and v_bbo.ask is not None and v_bbo.ask > 0:
                    price, size_at_best, venue = v_bbo.ask, v_bbo.ask_size, target_venue
                else:
                    venue_fallback += 1   # venue had no quote → fall back to NBBO

            if price is None:
                snap = self.lookup.at(c.timestamp_ns)
                if parent.side == "sell":
                    price = snap.nbb
                    size_at_best = snap.nbb_size
                    venue = snap.nbb_venue
                else:
                    price = snap.nbo
                    size_at_best = snap.nbo_size
                    venue = snap.nbo_venue

            if price is None:
                null_nbbo += 1
                continue
            if size_at_best is not None and c.quantity > size_at_best:
                oversize += 1

            fill_rows.append({
                "timestamp_ns": c.timestamp_ns,
                "price": float(price),
                "quantity": int(c.quantity),
                "venue": str(venue) if venue else "",
                "side": parent.side,
            })

        if not fill_rows:
            raise ValueError(
                f"No NBBO available at any child timestamp. "
                f"Data coverage problem; abort."
            )

        fills = pl.DataFrame(fill_rows).sort("timestamp_ns")

        # Market VWAP over the actual execution window
        market_vwap = compute_market_vwap(
            self.df, start_ns=parent.start_ns, end_ns=parent.end_ns
        )

        # Build planned schedule (cumulative qty by ts) for schedule_deviation
        planned = strategy.planned_cumulative(children)

        metrics = compute_all_metrics(
            fills, self.df, self.lookup,
            side=parent.side,
            arrival_mid=arrival,
            window_start_ns=parent.start_ns,
            window_end_ns=parent.end_ns,
            planned_schedule=planned,
        )

        return BacktestResult(
            parent=parent,
            strategy_name=strategy.name,
            fills=fills,
            metrics=metrics,
            arrival_mid=arrival,
            market_vwap=market_vwap,
            oversize_count=oversize,
            null_nbbo_count=null_nbbo,
        )
