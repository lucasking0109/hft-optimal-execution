"""Per-venue execution-quality metrics for SOR analysis.

We compute four metrics per venue (and per hour where useful):
    1. nbbo_share_pct       — fraction of NBBO updates this venue sets
    2. volume_share_pct     — fraction of trade volume this venue absorbs
    3. avg_depth_at_best    — average quote size at venue's local BBO
    4. adverse_selection_bps — Lee-Ready signed markout @ 60s after fills

Routable venue note: FINRA isn't an exchange — it's where off-exchange (dark
pool / internalizer) trades print. We compute FINRA metrics for awareness
but flag it as non-routable downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from hft.analysis.nbbo_lookup import NBBOLookup
from hft.data.timeparse import add_eq_ns_of_day, filter_rth

# Exchanges we treat as "routable" (i.e. ones a smart-router could send to)
NON_ROUTABLE_VENUES: set[str] = {"FINRA"}


@dataclass
class VenueMetrics:
    venue: str
    nbbo_share_bid_pct: float       # % of QUOTE BID NB rows from this venue
    nbbo_share_ask_pct: float       # % of QUOTE ASK NB rows from this venue
    volume_share_pct: float
    avg_depth_at_best: float        # avg quote size at venue's BBO (shares)
    adverse_selection_bps: float    # signed 60s markout, weighted by qty
    n_trades: int
    routable: bool


# ---------------------------------------------------------------------------
# Per-venue computations
# ---------------------------------------------------------------------------

def compute_venue_nbbo_share(df: pl.DataFrame) -> pl.DataFrame:
    """Per-venue % of NBBO updates (split into bid/ask sides)."""
    nb_bid = df.filter(pl.col("EventType") == "QUOTE BID NB")
    nb_ask = df.filter(pl.col("EventType") == "QUOTE ASK NB")

    bid_total = max(1, len(nb_bid))
    ask_total = max(1, len(nb_ask))

    bid_share = (
        nb_bid.group_by("Exchange").agg(pl.len().alias("n"))
        .with_columns((pl.col("n") / bid_total * 100).alias("nbbo_share_bid_pct"))
        .select("Exchange", "nbbo_share_bid_pct")
    )
    ask_share = (
        nb_ask.group_by("Exchange").agg(pl.len().alias("n"))
        .with_columns((pl.col("n") / ask_total * 100).alias("nbbo_share_ask_pct"))
        .select("Exchange", "nbbo_share_ask_pct")
    )
    out = bid_share.join(ask_share, on="Exchange", how="full", coalesce=True)
    return out.with_columns(
        pl.col("nbbo_share_bid_pct").fill_null(0.0),
        pl.col("nbbo_share_ask_pct").fill_null(0.0),
    )


def compute_venue_volume_share(df: pl.DataFrame) -> pl.DataFrame:
    """Per-venue trade-volume share."""
    trades = df.filter(pl.col("EventType").is_in(["TRADE", "TRADE NB"]))
    if trades.is_empty():
        return pl.DataFrame(schema={"Exchange": pl.Utf8, "volume_share_pct": pl.Float64,
                                    "n_trades": pl.Int64})

    total_vol = int(trades["Quantity"].sum())
    if total_vol <= 0:
        raise ValueError("Zero total volume — cannot compute share")

    return (
        trades.group_by("Exchange")
        .agg(
            pl.col("Quantity").sum().alias("vol"),
            pl.len().alias("n_trades"),
        )
        .with_columns((pl.col("vol") / total_vol * 100).alias("volume_share_pct"))
        .select("Exchange", "volume_share_pct", "n_trades")
    )


def compute_venue_depth(df: pl.DataFrame) -> pl.DataFrame:
    """Average quoted size at each venue's local BBO."""
    quotes = df.filter(
        pl.col("EventType").is_in(["QUOTE BID", "QUOTE ASK"])
        & (pl.col("Quantity") > 0) & (pl.col("Price") > 0)
    )
    if quotes.is_empty():
        return pl.DataFrame(schema={"Exchange": pl.Utf8, "avg_depth_at_best": pl.Float64})
    return (
        quotes.group_by("Exchange")
        .agg(pl.col("Quantity").mean().alias("avg_depth_at_best"))
    )


def compute_venue_adverse_selection(
    df: pl.DataFrame,
    *,
    horizon_seconds: int = 60,
    rth_only: bool = True,
) -> pl.DataFrame:
    """Per-venue adverse selection cost in bps.

    For each TRADE event:
        1. Determine sign via Lee-Ready: trade_px vs prevailing NBBO mid
            >mid → buy aggressor (+1); <mid → sell (-1); ==mid → skip
        2. Look up mid at fill_time + horizon_seconds
        3. cost_bps = side × (mid_after − fill_price) / fill_price × 1e4
    Average across each venue (weighted by qty).

    Higher value → trades on this venue are typically followed by adverse
    moves (informed flow goes through here).
    """
    if "ns_of_day" not in df.columns:
        df = add_eq_ns_of_day(df)
    if rth_only:
        df = filter_rth(df, src="Timestamp")

    lookup = NBBOLookup(df)
    horizon_ns = horizon_seconds * 1_000_000_000

    trades = df.filter(pl.col("EventType").is_in(["TRADE", "TRADE NB"]))
    if trades.is_empty():
        return pl.DataFrame(schema={"Exchange": pl.Utf8, "adverse_selection_bps": pl.Float64})

    weighted_cost: dict[str, float] = {}
    weighted_qty: dict[str, int] = {}

    for row in trades.iter_rows(named=True):
        ns = row["ns_of_day"]
        venue = str(row["Exchange"])
        qty = row["Quantity"]
        px = row["Price"]
        if qty is None or qty <= 0 or px is None or px <= 0:
            continue

        snap_now = lookup.at(ns)
        mid_now = snap_now.mid
        mid_after = lookup.mid_at(ns + horizon_ns)
        if mid_now is None or mid_after is None or mid_now <= 0 or mid_after <= 0:
            continue

        # Lee-Ready
        if px > mid_now * (1 + 1e-9):
            side = +1
        elif px < mid_now * (1 - 1e-9):
            side = -1
        else:
            continue

        cost = side * (mid_after - px) / px * 10000
        weighted_cost[venue] = weighted_cost.get(venue, 0.0) + cost * qty
        weighted_qty[venue] = weighted_qty.get(venue, 0) + qty

    rows = [
        {"Exchange": v, "adverse_selection_bps": weighted_cost[v] / weighted_qty[v]}
        for v in weighted_cost if weighted_qty[v] > 0
    ]
    return pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={"Exchange": pl.Utf8, "adverse_selection_bps": pl.Float64}
    )


def _cast_exchange_to_utf8(df: pl.DataFrame) -> pl.DataFrame:
    """Categorical → Utf8 so joins across DataFrames don't choke."""
    if "Exchange" in df.columns and df.schema["Exchange"] != pl.Utf8:
        return df.with_columns(pl.col("Exchange").cast(pl.Utf8))
    return df


def compute_all_venue_metrics(
    df: pl.DataFrame,
    *,
    horizon_seconds: int = 60,
    rth_only: bool = True,
) -> pl.DataFrame:
    """One-shot: returns DataFrame with one row per venue, all metrics joined."""
    if rth_only:
        df_for_summary = filter_rth(add_eq_ns_of_day(df), src="Timestamp")
    else:
        df_for_summary = add_eq_ns_of_day(df)

    nbbo = _cast_exchange_to_utf8(compute_venue_nbbo_share(df_for_summary))
    vol = _cast_exchange_to_utf8(compute_venue_volume_share(df_for_summary))
    depth = _cast_exchange_to_utf8(compute_venue_depth(df_for_summary))
    adv = _cast_exchange_to_utf8(
        compute_venue_adverse_selection(df_for_summary, horizon_seconds=horizon_seconds,
                                        rth_only=False)
    )

    out = (
        nbbo.join(vol, on="Exchange", how="full", coalesce=True)
        .join(depth, on="Exchange", how="full", coalesce=True)
        .join(adv, on="Exchange", how="full", coalesce=True)
        .with_columns([
            pl.col("nbbo_share_bid_pct").fill_null(0.0),
            pl.col("nbbo_share_ask_pct").fill_null(0.0),
            pl.col("volume_share_pct").fill_null(0.0),
            pl.col("n_trades").fill_null(0),
            pl.col("avg_depth_at_best").fill_null(0.0),
            pl.col("adverse_selection_bps").fill_null(0.0),
            pl.col("Exchange").is_in(list(NON_ROUTABLE_VENUES)).not_().alias("routable"),
        ])
        .sort("volume_share_pct", descending=True)
    )
    return out


# ---------------------------------------------------------------------------
# Hourly per-venue NBBO share (for the heatmap visualisation)
# ---------------------------------------------------------------------------

def compute_venue_nbbo_share_hourly(df: pl.DataFrame, *, rth_only: bool = True) -> pl.DataFrame:
    """Hour × Venue grid of NBBO bid+ask share."""
    if "ns_of_day" not in df.columns:
        df = add_eq_ns_of_day(df)
    if rth_only:
        df = filter_rth(df, src="Timestamp")

    nb = df.filter(pl.col("EventType").is_in(["QUOTE BID NB", "QUOTE ASK NB"]))
    if nb.is_empty():
        return pl.DataFrame(schema={"hour_et": pl.Int64, "Exchange": pl.Utf8, "share_pct": pl.Float64})
    nb = nb.with_columns((pl.col("ns_of_day") // (3600 * 1_000_000_000)).alias("hour_et"))
    per_hour_total = nb.group_by("hour_et").agg(pl.len().alias("total"))
    per_hour_venue = nb.group_by(["hour_et", "Exchange"]).agg(pl.len().alias("n"))
    return (
        per_hour_venue.join(per_hour_total, on="hour_et")
        .with_columns((pl.col("n") / pl.col("total") * 100).alias("share_pct"))
        .select("hour_et", "Exchange", "share_pct")
        .sort(["hour_et", "share_pct"], descending=[False, True])
    )
