"""Fast NBBO lookup at arbitrary timestamps.

Uses the venue-published `QUOTE BID NB` / `QUOTE ASK NB` rows already in
the TAQ data so we don't repeat per-venue NBBO reconstruction. For each
query timestamp we forward-fill the most recent NB row.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import polars as pl

from hft.data.timeparse import add_eq_ns_of_day


@dataclass
class NBBOSnapshot:
    nbb: float | None
    nbb_size: int | None
    nbb_venue: str | None
    nbo: float | None
    nbo_size: int | None
    nbo_venue: str | None

    @property
    def mid(self) -> float | None:
        if self.nbb is None or self.nbo is None:
            return None
        if self.nbb <= 0 or self.nbo <= 0:
            return None
        return (self.nbb + self.nbo) / 2.0

    @property
    def spread_bps(self) -> float | None:
        if self.mid is None:
            return None
        return (self.nbo - self.nbb) / self.mid * 10000


class NBBOLookup:
    """Build once for a (ticker, date), then call .at(ns) any number of times.

    Internally stores parallel arrays of (ns, bid, bid_size, bid_venue) and
    (ns, ask, ask_size, ask_venue). `at(ns)` does a binary search on each.
    """

    def __init__(self, df: pl.DataFrame):
        if "ns_of_day" not in df.columns:
            df = add_eq_ns_of_day(df)

        bid = (
            df.filter(pl.col("EventType") == "QUOTE BID NB")
            .select("ns_of_day", "Price", "Quantity", "Exchange")
            .sort("ns_of_day")
        )
        ask = (
            df.filter(pl.col("EventType") == "QUOTE ASK NB")
            .select("ns_of_day", "Price", "Quantity", "Exchange")
            .sort("ns_of_day")
        )

        self._bid_ns = bid["ns_of_day"].to_list()
        self._bid_px = bid["Price"].to_list()
        self._bid_sz = bid["Quantity"].to_list()
        self._bid_vn = [str(v) for v in bid["Exchange"].to_list()]

        self._ask_ns = ask["ns_of_day"].to_list()
        self._ask_px = ask["Price"].to_list()
        self._ask_sz = ask["Quantity"].to_list()
        self._ask_vn = [str(v) for v in ask["Exchange"].to_list()]

    def _lookup(self, ns_arr, px_arr, sz_arr, vn_arr, ns: int):
        if not ns_arr:
            return None, None, None
        idx = bisect.bisect_right(ns_arr, ns) - 1
        if idx < 0:
            return None, None, None
        return px_arr[idx], sz_arr[idx], vn_arr[idx]

    def at(self, ns: int) -> NBBOSnapshot:
        b_px, b_sz, b_vn = self._lookup(self._bid_ns, self._bid_px, self._bid_sz, self._bid_vn, ns)
        a_px, a_sz, a_vn = self._lookup(self._ask_ns, self._ask_px, self._ask_sz, self._ask_vn, ns)
        return NBBOSnapshot(
            nbb=b_px if (b_px is not None and b_px > 0) else None,
            nbb_size=b_sz if (b_sz is not None and b_sz > 0) else None,
            nbb_venue=b_vn,
            nbo=a_px if (a_px is not None and a_px > 0) else None,
            nbo_size=a_sz if (a_sz is not None and a_sz > 0) else None,
            nbo_venue=a_vn,
        )

    def mid_at(self, ns: int) -> float | None:
        return self.at(ns).mid


@dataclass
class VenueBBO:
    """A single venue's best bid/ask at a given timestamp."""
    bid: float | None
    bid_size: int | None
    ask: float | None
    ask_size: int | None

    @property
    def mid(self) -> float | None:
        if self.bid is None or self.ask is None or self.bid <= 0 or self.ask <= 0:
            return None
        return (self.bid + self.ask) / 2.0


class VenueBBOLookup:
    """Per-venue best bid/ask lookup at arbitrary timestamps.

    Uses raw `QUOTE BID` / `QUOTE ASK` rows (NOT the `NB` aggregated rows),
    grouping by `Exchange` venue. Each venue gets its own sorted timeline;
    `at(ns)` returns dict[venue → VenueBBO] via per-venue binary search +
    forward-fill.

    Used by SOR routing strategies (Phase C) to fill child orders at a
    specific venue's quote (rather than NBBO).
    """

    def __init__(self, df: pl.DataFrame):
        if "ns_of_day" not in df.columns:
            df = add_eq_ns_of_day(df)

        bid_df = (
            df.filter(pl.col("EventType") == "QUOTE BID")
            .select("ns_of_day", "Price", "Quantity", "Exchange")
            .sort("ns_of_day")
        )
        ask_df = (
            df.filter(pl.col("EventType") == "QUOTE ASK")
            .select("ns_of_day", "Price", "Quantity", "Exchange")
            .sort("ns_of_day")
        )

        # Group by venue → sorted timeline
        self._venues: list[str] = sorted(set(
            [str(v) for v in bid_df["Exchange"].to_list()]
            + [str(v) for v in ask_df["Exchange"].to_list()]
        ))

        # Per-venue (ns, px, sz) lists
        self._bid_by_venue: dict[str, tuple[list[int], list[float], list[int]]] = {}
        self._ask_by_venue: dict[str, tuple[list[int], list[float], list[int]]] = {}
        for v in self._venues:
            bv = bid_df.filter(pl.col("Exchange") == v)
            av = ask_df.filter(pl.col("Exchange") == v)
            self._bid_by_venue[v] = (
                bv["ns_of_day"].to_list(),
                bv["Price"].to_list(),
                bv["Quantity"].to_list(),
            )
            self._ask_by_venue[v] = (
                av["ns_of_day"].to_list(),
                av["Price"].to_list(),
                av["Quantity"].to_list(),
            )

    @property
    def venues(self) -> list[str]:
        return list(self._venues)

    def _lookup_one(self, timeline: tuple[list, list, list], ns: int):
        ns_arr, px_arr, sz_arr = timeline
        if not ns_arr:
            return None, None
        idx = bisect.bisect_right(ns_arr, ns) - 1
        if idx < 0:
            return None, None
        px = px_arr[idx]
        sz = sz_arr[idx]
        return (px if px > 0 else None), (sz if sz > 0 else None)

    def at(self, ns: int) -> dict[str, VenueBBO]:
        """Return dict mapping venue → VenueBBO at timestamp ns."""
        out: dict[str, VenueBBO] = {}
        for v in self._venues:
            bid_px, bid_sz = self._lookup_one(self._bid_by_venue[v], ns)
            ask_px, ask_sz = self._lookup_one(self._ask_by_venue[v], ns)
            out[v] = VenueBBO(bid=bid_px, bid_size=bid_sz, ask=ask_px, ask_size=ask_sz)
        return out

    def at_venue(self, ns: int, venue: str) -> VenueBBO:
        """Return BBO for a specific venue at ns. Raises if venue not found."""
        if venue not in self._bid_by_venue:
            raise KeyError(f"Venue {venue!r} not found. Available: {self.venues}")
        bid_px, bid_sz = self._lookup_one(self._bid_by_venue[venue], ns)
        ask_px, ask_sz = self._lookup_one(self._ask_by_venue[venue], ns)
        return VenueBBO(bid=bid_px, bid_size=bid_sz, ask=ask_px, ask_size=ask_sz)
