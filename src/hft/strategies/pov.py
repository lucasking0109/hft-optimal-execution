"""POV (Percent of Volume) execution strategy.

Maintains a target % of market volume per time bucket. Distinct from
VWAP-following:
  - VWAP-following: parent_qty distributed proportional to volume share —
    matches market VWAP by construction (no per-bucket cap).
  - POV: per-bucket child_qty = min(target_pov × bucket_volume,
    cap_pov × bucket_volume). Provides "stealth" execution — if market
    is quiet, you trade less.

If `target_pov` is None it is auto-derived as `parent_qty / total_expected_volume`,
giving a TWAP-equivalent if no cap is hit. The interesting case is when
`cap_pov` binds: orders are deferred or unfinished.

For Phase 2/3-style backtest the parent window is fixed. When `cap_pov`
binds (`parent.quantity > cap_pov × total_volume`), `force_completion=True`
distributes the residual via **water-fill** — proportional to each bucket's
volume — so every child's effective POV inflates uniformly above the cap
rather than the residual being concentrated in the final bucket.

Set `force_completion=False` to allow under-fill instead when cap binds.
"""

from __future__ import annotations

import polars as pl

from hft.strategies.base import ChildOrder, ExecutionStrategy, ParentOrder

NS_PER_MIN = 60 * 1_000_000_000


class POVStrategy(ExecutionStrategy):
    """Percent of Volume strategy with optional cap and auto target."""

    def __init__(
        self,
        *,
        target_pov: float | None = None,
        cap_pov: float = 0.20,
        force_completion: bool = True,
    ):
        if cap_pov <= 0 or cap_pov > 1:
            raise ValueError(f"cap_pov must be in (0, 1], got {cap_pov}")
        if target_pov is not None and (target_pov <= 0 or target_pov > 1):
            raise ValueError(f"target_pov must be in (0, 1], got {target_pov}")
        self.target_pov = target_pov
        self.cap_pov = cap_pov
        self.force_completion = force_completion
        eff = target_pov if target_pov is not None else "auto"
        self.name = f"pov_{eff}_cap{cap_pov}"

    def schedule(
        self,
        parent: ParentOrder,
        *,
        market_context: dict,
    ) -> list[ChildOrder]:
        profile: pl.DataFrame | None = market_context.get("volume_profile")
        if profile is None or profile.is_empty():
            raise ValueError(
                "volume_profile DataFrame missing in market_context. "
                "Provide compute_volume_profile(...) result."
            )
        bin_minutes: int = market_context.get("bin_minutes", 5)
        bin_ns = bin_minutes * NS_PER_MIN

        # Filter buckets overlapping the execution window
        bucket_start_ns = profile["bucket_min"].to_list()
        bucket_qty = profile["volume"].to_list()
        eligible = []  # (ns_mid, expected_volume_in_overlap)
        for b, vol in zip(bucket_start_ns, bucket_qty):
            ns_start = int(b) * bin_ns
            ns_end = ns_start + bin_ns
            if ns_start < parent.end_ns and ns_end > parent.start_ns:
                overlap_start = max(ns_start, parent.start_ns)
                overlap_end = min(ns_end, parent.end_ns)
                overlap_frac = (overlap_end - overlap_start) / bin_ns
                ts_mid = (overlap_start + overlap_end) // 2
                eligible.append((ts_mid, float(vol) * overlap_frac))

        if not eligible:
            raise ValueError(
                f"No volume buckets overlap window [{parent.start_ns}, {parent.end_ns}]"
            )

        total_vol = sum(v for _, v in eligible)
        if total_vol <= 0:
            raise ValueError("Volume profile sums to 0")

        # Determine target_pov
        target = self.target_pov
        if target is None:
            target = parent.quantity / total_vol
            if target > self.cap_pov:
                # Auto-target would exceed cap → set to cap (will under-fill)
                target = self.cap_pov

        # Allocate per bucket: min(target × vol, cap × vol) = min(target, cap) × vol.
        # Track (ts, vol, qty) per bucket so we can water-fill afterwards.
        per_bucket_pov = min(target, self.cap_pov)
        per_bucket: list[tuple[int, float, int]] = []  # (ts, vol, qty)
        allocated = 0
        for ts, vol in eligible:
            qty = int(round(per_bucket_pov * vol))
            if qty <= 0:
                continue
            per_bucket.append((ts, vol, qty))
            allocated += qty

        if self.force_completion and allocated != parent.quantity:
            if not per_bucket:
                # No bucket survived sizing — emit one at midpoint with full qty
                ts_mid = (parent.start_ns + parent.end_ns) // 2
                return [ChildOrder(timestamp_ns=ts_mid, quantity=parent.quantity)]

            residual = parent.quantity - allocated
            # Water-fill: distribute residual proportional to bucket volume.
            # If cap binds, every child's effective POV inflates by the same
            # multiplicative factor (= parent_qty / allocated_at_cap),
            # rather than concentrating residual in a single bucket.
            vol_sum = sum(v for _, v, _ in per_bucket)
            if vol_sum <= 0:
                # Defensive: split residual evenly across children
                share = residual // len(per_bucket)
                rem = residual - share * len(per_bucket)
                new_per_bucket = [
                    (ts, vol, qty + share + (rem if i == len(per_bucket) - 1 else 0))
                    for i, (ts, vol, qty) in enumerate(per_bucket)
                ]
            else:
                # Volume-proportional water-fill; correct rounding drift on the last child.
                add_alloc = 0
                new_per_bucket = []
                for i, (ts, vol, qty) in enumerate(per_bucket):
                    if i == len(per_bucket) - 1:
                        add = residual - add_alloc  # residue → last
                    else:
                        add = int(round(residual * (vol / vol_sum)))
                        add_alloc += add
                    new_per_bucket.append((ts, vol, qty + add))
            per_bucket = new_per_bucket

        return [
            ChildOrder(timestamp_ns=ts, quantity=max(1, qty))
            for ts, _, qty in per_bucket
        ]
