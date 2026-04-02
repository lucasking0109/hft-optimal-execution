"""SORRoutingStrategy — Phase C wrapper that routes child orders to venues.

Wraps any base ExecutionStrategy and assigns a `target_venue` to each child
order based on a venue allocation dict. The backtest engine fills at that
venue's BBO instead of NBBO when `target_venue` is set.

Three routing modes:
  - `'top1'`     : send all children to the single highest-scored venue
  - `'weighted'` : random per-child venue draw weighted by allocation
  - `'naive'`    : random per-child venue draw weighted by historical volume

Works with any base strategy (TWAP / VWAP-following / AC / etc.).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from hft.strategies.base import ChildOrder, ExecutionStrategy, ParentOrder

RoutingMode = Literal["top1", "weighted", "naive"]


class SORRoutingStrategy(ExecutionStrategy):
    """Wrap a base execution strategy with venue-routing logic."""

    def __init__(
        self,
        base_strategy: ExecutionStrategy,
        venue_allocation: dict[str, float],
        *,
        mode: RoutingMode = "top1",
        seed: int | None = None,
    ):
        if not venue_allocation:
            raise ValueError("venue_allocation cannot be empty")
        if mode not in ("top1", "weighted", "naive"):
            raise ValueError(f"mode must be top1/weighted/naive, got {mode}")
        self.base_strategy = base_strategy
        # Normalize allocation weights
        total = sum(venue_allocation.values())
        if total <= 0:
            raise ValueError(f"venue_allocation weights must sum > 0, got {total}")
        self.venue_allocation = {v: w / total for v, w in venue_allocation.items()}
        self.mode = mode
        self._rng = np.random.default_rng(seed)
        # Cache top-1 venue for top1 mode
        self._top1_venue = max(self.venue_allocation, key=self.venue_allocation.get)
        self.name = f"sor_{mode}_{base_strategy.name}"

    def _pick_venue(self) -> str:
        if self.mode == "top1":
            return self._top1_venue
        # weighted / naive: random draw weighted by allocation
        venues = list(self.venue_allocation.keys())
        weights = np.array([self.venue_allocation[v] for v in venues])
        idx = int(self._rng.choice(len(venues), p=weights))
        return venues[idx]

    def schedule(
        self,
        parent: ParentOrder,
        *,
        market_context: dict,
    ) -> list[ChildOrder]:
        """Get base schedule, then annotate each child with target_venue."""
        base_children = self.base_strategy.schedule(parent, market_context=market_context)
        return [
            ChildOrder(
                timestamp_ns=c.timestamp_ns,
                quantity=c.quantity,
                target_venue=self._pick_venue(),
            )
            for c in base_children
        ]
