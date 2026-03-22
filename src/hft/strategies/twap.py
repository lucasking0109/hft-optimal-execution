"""TWAP — Time-Weighted Average Price strategy.

Splits the parent quantity evenly across N equally-spaced child orders.
"""

from __future__ import annotations

from hft.strategies.base import ChildOrder, ExecutionStrategy, ParentOrder


class TWAPStrategy(ExecutionStrategy):
    name = "twap"

    def __init__(self, num_slices: int = 60):
        if num_slices <= 0:
            raise ValueError(f"num_slices must be > 0, got {num_slices}")
        self.num_slices = num_slices

    def schedule(self, parent: ParentOrder, *, market_context: dict) -> list[ChildOrder]:
        total_qty = parent.quantity
        N = self.num_slices

        # Distribute quantity as evenly as possible (handle remainder)
        base = total_qty // N
        remainder = total_qty - base * N
        slice_qtys = [base + (1 if i < remainder else 0) for i in range(N)]

        # Equally spaced timestamps
        duration = parent.end_ns - parent.start_ns
        # Place slice i at start_ns + (i + 0.5) * duration / N (centre of bucket)
        timestamps = [
            parent.start_ns + int((i + 0.5) * duration / N) for i in range(N)
        ]

        return [ChildOrder(timestamp_ns=ts, quantity=q)
                for ts, q in zip(timestamps, slice_qtys) if q > 0]
