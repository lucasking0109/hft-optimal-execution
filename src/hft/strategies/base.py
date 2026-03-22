"""ExecutionStrategy ABC and core dataclasses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import polars as pl

Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class ParentOrder:
    """Top-level liquidation/accumulation request."""

    ticker: str
    date: str
    side: Side
    quantity: int  # total shares to execute (>0)
    start_ns: int  # nanoseconds-of-day when the order is initiated
    end_ns: int    # deadline by which it must finish

    def __post_init__(self):
        if self.quantity <= 0:
            raise ValueError(f"quantity must be > 0, got {self.quantity}")
        if self.end_ns <= self.start_ns:
            raise ValueError("end_ns must be after start_ns")


@dataclass(frozen=True)
class ChildOrder:
    """A single market-order slice, scheduled for a specific time.

    `target_venue` is set by Phase C SOR routing strategies when present;
    backtest engine fills at that specific venue's BBO instead of NBBO.
    Default None preserves backward compatibility with non-routed strategies.
    """

    timestamp_ns: int
    quantity: int
    target_venue: str | None = None

    def __post_init__(self):
        if self.quantity <= 0:
            raise ValueError(f"quantity must be > 0, got {self.quantity}")


@dataclass(frozen=True)
class Fill:
    timestamp_ns: int
    price: float
    quantity: int
    venue: str
    side: Side


class ExecutionStrategy(ABC):
    """Strategies decide when and how much to slice the parent into children.

    For Phase 2 we only support **offline** scheduling: the strategy returns
    the full child order list up front. Phase 6 RL strategies will override
    a more granular `step()` API so they can react to live state. We will
    refactor when we get there — keeping the base lean for now.
    """

    name: str = "base"

    @abstractmethod
    def schedule(self, parent: ParentOrder, *, market_context: dict) -> list[ChildOrder]:
        """Return ordered list of child orders summing to parent.quantity.

        market_context can include: volume_profile_df, NBBOLookup, etc.,
        depending on the strategy's needs.
        """

    def planned_cumulative(self, children: list[ChildOrder]) -> list[tuple[int, int]]:
        """Return [(timestamp_ns, cumulative_qty), ...] for schedule_deviation metric."""
        cum = 0
        out = []
        for c in children:
            cum += c.quantity
            out.append((c.timestamp_ns, cum))
        return out
