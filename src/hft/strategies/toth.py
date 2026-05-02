"""Tóth square-root impact-aware execution strategy.

Reference: Tóth et al. (2011), "Anomalous price impact and the critical nature
of liquidity in financial markets" — empirical impact ≈ Y · σ · √(Q/V).

Under risk-neutral assumption, the cost-minimizing schedule under √-impact
is q_t ∝ V_t (proportional to volume), same as VWAP-following at the limit.
The practical differentiator is **participation rate capping**: Tóth-aware
desks keep per-bucket child sizes small to limit non-linear impact.

This implementation:
  1. Start with volume-proportional allocation (VWAP-following baseline).
  2. If any child exceeds `participation_cap × bucket_volume`, clip it.
  3. Redistribute the residual to remaining under-cap buckets, weighted by
     headroom (water-filling). Iterate until no cap is exceeded or all
     buckets are saturated.
  4. If still residual, stuff into the largest-volume bucket as a final
     fallback (loud caveat: this child violates Tóth participation discipline).

Differs from POV:
  - POV stuffs residual at last child (TIME priority).
  - Tóth water-fills residual proportional to remaining headroom (VOLUME priority).

Y and sigma are stored for potential cost-quote computation but do not
affect the schedule under risk-neutral assumption.
"""

from __future__ import annotations

import polars as pl

from hft.strategies.base import ChildOrder, ExecutionStrategy, ParentOrder

NS_PER_MIN = 60 * 1_000_000_000


class TothStrategy(ExecutionStrategy):
    """Tóth-style schedule with participation cap + water-filling residual."""

    def __init__(
        self,
        *,
        participation_cap: float = 0.10,
        Y: float = 1.0,
        sigma_bps: float = 10.0,
        max_iterations: int = 20,
    ):
        if not (0 < participation_cap <= 1.0):
            raise ValueError(f"participation_cap must be in (0,1], got {participation_cap}")
        if Y <= 0:
            raise ValueError(f"Y must be > 0, got {Y}")
        if sigma_bps <= 0:
            raise ValueError(f"sigma_bps must be > 0, got {sigma_bps}")
        self.participation_cap = participation_cap
        self.Y = Y
        self.sigma_bps = sigma_bps
        self.max_iterations = max_iterations
        self.name = f"toth_pcap{participation_cap}"

    def expected_cost_bps(self, parent_qty: int, total_volume: float) -> float:
        """Tóth √-impact estimated cost for the parent order (informational)."""
        if total_volume <= 0:
            return float("inf")
        import math
        return self.Y * self.sigma_bps * math.sqrt(parent_qty / total_volume)

    def schedule(
        self,
        parent: ParentOrder,
        *,
        market_context: dict,
    ) -> list[ChildOrder]:
        profile: pl.DataFrame | None = market_context.get("volume_profile")
        if profile is None or profile.is_empty():
            raise ValueError("volume_profile DataFrame missing in market_context")
        bin_minutes: int = market_context.get("bin_minutes", 5)
        bin_ns = bin_minutes * NS_PER_MIN

        # Filter buckets overlapping window
        eligible = []   # (ts_mid, expected_volume_in_overlap)
        for b, vol in zip(profile["bucket_min"].to_list(),
                          profile["volume"].to_list()):
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
                f"No volume buckets overlap [{parent.start_ns}, {parent.end_ns}]"
            )

        n = len(eligible)
        timestamps = [ts for ts, _ in eligible]
        volumes = [v for _, v in eligible]
        total_vol = sum(volumes)
        if total_vol <= 0:
            raise ValueError("Volume profile sums to 0")

        target_qty = parent.quantity

        # Step 1: VWAP-proportional baseline
        qtys = [target_qty * v / total_vol for v in volumes]

        # Step 2-3: water-filling cap iteration
        for _ in range(self.max_iterations):
            caps = [v * self.participation_cap for v in volumes]
            over = [max(0.0, qtys[i] - caps[i]) for i in range(n)]
            total_over = sum(over)
            if total_over < 1.0:
                break
            # Clip overs
            for i in range(n):
                qtys[i] = min(qtys[i], caps[i])
            # Redistribute residual to buckets with headroom (cap - current)
            headroom = [max(0.0, caps[i] - qtys[i]) for i in range(n)]
            total_head = sum(headroom)
            if total_head < 1.0:
                # All buckets saturated; stuff residual into largest-volume bucket
                if total_over > 0:
                    biggest = max(range(n), key=lambda i: volumes[i])
                    qtys[biggest] += total_over
                break
            for i in range(n):
                qtys[i] += total_over * headroom[i] / total_head

        # Convert to integers preserving total
        int_qtys = [int(round(q)) for q in qtys]
        diff = target_qty - sum(int_qtys)
        if diff != 0:
            idx = max(range(n), key=lambda i: int_qtys[i])
            int_qtys[idx] += diff

        children = [
            ChildOrder(timestamp_ns=ts, quantity=q)
            for ts, q in zip(timestamps, int_qtys) if q > 0
        ]
        return children
