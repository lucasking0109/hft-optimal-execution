"""VWAP-Following strategy — slice quantity in proportion to historical
volume profile so the schedule mirrors typical intraday volume.

For Phase 2 baseline we accept the volume profile of the **same trading day**
as input (look-ahead bias acknowledged); Phase 3+ can use rolling estimate
from prior days.
"""

from __future__ import annotations

import polars as pl

from hft.strategies.base import ChildOrder, ExecutionStrategy, ParentOrder

NS_PER_MIN = 60 * 1_000_000_000


class VWAPFollowingStrategy(ExecutionStrategy):
    name = "vwap_following"

    def __init__(self, num_slices: int = 60):
        self.num_slices = num_slices

    def schedule(self, parent: ParentOrder, *, market_context: dict) -> list[ChildOrder]:
        profile: pl.DataFrame | None = market_context.get("volume_profile")
        if profile is None or profile.is_empty():
            raise ValueError(
                "volume_profile DataFrame missing in market_context. "
                "Provide compute_volume_profile(...) result. "
                "(NO silent fallback to TWAP — caller decides.)"
            )

        bin_minutes: int = market_context.get("bin_minutes", 5)

        # Filter profile rows whose bucket falls within [start_ns, end_ns]
        bin_ns = bin_minutes * NS_PER_MIN
        bucket_start_ns = profile["bucket_min"].to_list()
        bucket_qty = profile["volume"].to_list()
        # bucket_min is minutes-since-midnight / bin_minutes
        # Convert to ns_of_day
        eligible = []
        for b, vol in zip(bucket_start_ns, bucket_qty):
            ns_start = int(b) * bin_ns
            ns_end = ns_start + bin_ns
            # Bucket overlaps execution window if ns_start < parent.end_ns and ns_end > parent.start_ns
            if ns_start < parent.end_ns and ns_end > parent.start_ns:
                # Overlap fraction (in case bucket only partially in window)
                overlap_start = max(ns_start, parent.start_ns)
                overlap_end = min(ns_end, parent.end_ns)
                overlap_frac = (overlap_end - overlap_start) / bin_ns
                eligible.append((b, vol * overlap_frac, overlap_start, overlap_end))

        if not eligible:
            raise ValueError(
                f"No volume buckets overlap execution window [{parent.start_ns}, {parent.end_ns}]. "
                f"Check bin_minutes / window."
            )

        total_vol = sum(v for _, v, _, _ in eligible)
        if total_vol <= 0:
            raise ValueError("Volume profile sums to 0 — bad input")

        # Allocate parent quantity proportional to volume share
        target_qty = parent.quantity
        children = []
        allocated = 0
        for i, (_, vol, s, e) in enumerate(eligible):
            if i == len(eligible) - 1:
                qty = target_qty - allocated  # last bucket absorbs rounding
            else:
                qty = int(round(target_qty * vol / total_vol))
            if qty <= 0:
                continue
            # Place child at midpoint of overlapping window
            ts = (s + e) // 2
            children.append(ChildOrder(timestamp_ns=ts, quantity=qty))
            allocated += qty

        # Sanity check: total allocated should equal parent.quantity
        total_alloc = sum(c.quantity for c in children)
        if total_alloc != parent.quantity:
            # Adjust last child for any rounding leftover
            if children:
                last = children[-1]
                children[-1] = ChildOrder(
                    timestamp_ns=last.timestamp_ns,
                    quantity=last.quantity + (parent.quantity - total_alloc),
                )

        return children
