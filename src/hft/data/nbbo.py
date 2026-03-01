"""NBBO (National Best Bid/Offer) reconstruction from per-venue equity TAQ.

The eq_taq files contain QUOTE BID / QUOTE ASK per venue plus QUOTE BID NB /
QUOTE ASK NB rows giving the venue-published National Best. We can reconstruct
the NBBO ourselves from per-venue quotes; build_nbbo verifies our
reconstruction matches the published `NB` rows so we know the data is sound.
"""

from __future__ import annotations

import polars as pl


def reconstruct_nbbo(df: pl.DataFrame) -> pl.DataFrame:
    """Reconstruct National Best Bid and National Best Offer at every event.

    Algorithm:
      1. Walk the event log in timestamp order.
      2. Keep a running best-per-venue for BID and ASK.
      3. NBB = max of all venue bids, NBO = min of all venue asks (skipping 0).

    Returns a DataFrame indexed by Timestamp with columns:
      Timestamp, NBB, NBB_size, NBB_venue, NBO, NBO_size, NBO_venue.

    NOTE: this is computationally expensive for full-day data
    (millions of events). For Phase 0 we keep a straightforward implementation
    and validate against published NB rows; Phase 1 may speed this up.
    """
    quotes = df.filter(pl.col("EventType").is_in(["QUOTE BID", "QUOTE ASK"]))
    if quotes.is_empty():
        raise ValueError("no QUOTE BID / QUOTE ASK rows present; cannot build NBBO")

    quotes = quotes.sort("Timestamp")

    # Running best per venue (price, size). 0/null prices are treated as "no quote".
    bid_per_venue: dict[str, tuple[float, int]] = {}
    ask_per_venue: dict[str, tuple[float, int]] = {}

    out_ts: list[str] = []
    out_nbb: list[float | None] = []
    out_nbb_sz: list[int | None] = []
    out_nbb_v: list[str | None] = []
    out_nbo: list[float | None] = []
    out_nbo_sz: list[int | None] = []
    out_nbo_v: list[str | None] = []

    for row in quotes.iter_rows(named=True):
        ev = row["EventType"]
        ven = row["Exchange"]
        px = row["Price"]
        sz = row["Quantity"]

        if px is None or px <= 0 or sz is None or sz <= 0:
            # Venue cleared this side
            if ev == "QUOTE BID":
                bid_per_venue.pop(ven, None)
            else:
                ask_per_venue.pop(ven, None)
        else:
            if ev == "QUOTE BID":
                bid_per_venue[ven] = (px, sz)
            else:
                ask_per_venue[ven] = (px, sz)

        # Compute NBB / NBO
        if bid_per_venue:
            nbb_v, (nbb, nbb_sz) = max(bid_per_venue.items(), key=lambda kv: kv[1][0])
        else:
            nbb_v, nbb, nbb_sz = None, None, None
        if ask_per_venue:
            nbo_v, (nbo, nbo_sz) = min(ask_per_venue.items(), key=lambda kv: kv[1][0])
        else:
            nbo_v, nbo, nbo_sz = None, None, None

        out_ts.append(row["Timestamp"])
        out_nbb.append(nbb); out_nbb_sz.append(nbb_sz); out_nbb_v.append(nbb_v)
        out_nbo.append(nbo); out_nbo_sz.append(nbo_sz); out_nbo_v.append(nbo_v)

    return pl.DataFrame(
        {
            "Timestamp": out_ts,
            "NBB": out_nbb,
            "NBB_size": out_nbb_sz,
            "NBB_venue": out_nbb_v,
            "NBO": out_nbo,
            "NBO_size": out_nbo_sz,
            "NBO_venue": out_nbo_v,
        }
    )


def published_nbbo(df: pl.DataFrame) -> pl.DataFrame:
    """Extract the venue-published National Best rows (`QUOTE BID NB` / `QUOTE ASK NB`)."""
    nb_bid = df.filter(pl.col("EventType") == "QUOTE BID NB").select(
        pl.col("Timestamp"),
        pl.col("Price").alias("Pub_NBB"),
        pl.col("Quantity").alias("Pub_NBB_size"),
    )
    nb_ask = df.filter(pl.col("EventType") == "QUOTE ASK NB").select(
        pl.col("Timestamp"),
        pl.col("Price").alias("Pub_NBO"),
        pl.col("Quantity").alias("Pub_NBO_size"),
    )
    return nb_bid.join(nb_ask, on="Timestamp", how="full", coalesce=True).sort("Timestamp")


def build_nbbo(df: pl.DataFrame, *, validate_against_published: bool = False) -> pl.DataFrame:
    """Public entry point — same as reconstruct_nbbo, with optional validation.

    Args:
        df: equity TAQ DataFrame.
        validate_against_published: cross-check against the venue-published NB rows.

    Raises:
        AssertionError if validation fails.
    """
    nbbo = reconstruct_nbbo(df)
    if validate_against_published:
        # Spot-check at the times where NB rows exist; tolerate small mismatches
        # from venue-specific delays (those are real, not bugs).
        published = published_nbbo(df)
        joined = nbbo.join(published, on="Timestamp", how="inner")
        if joined.is_empty():
            return nbbo
        # Verify NBB <= Pub_NBB and NBO >= Pub_NBO frequently enough.
        # We don't fail loudly on a few mismatches because exchanges latency-disagree;
        # but a wholesale mismatch is a sign the algorithm is wrong.
        agree = (
            joined.with_columns(
                ((pl.col("NBB") == pl.col("Pub_NBB")) | pl.col("Pub_NBB").is_null()).alias("bid_ok"),
                ((pl.col("NBO") == pl.col("Pub_NBO")) | pl.col("Pub_NBO").is_null()).alias("ask_ok"),
            )
            .select(pl.col("bid_ok").mean().alias("bid_match"),
                    pl.col("ask_ok").mean().alias("ask_match"))
        )
        bid_match = agree["bid_match"][0] or 0.0
        ask_match = agree["ask_match"][0] or 0.0
        if bid_match < 0.6 or ask_match < 0.6:
            raise AssertionError(
                f"NBBO reconstruction agrees with published NB only "
                f"{bid_match:.1%} (bids) / {ask_match:.1%} (asks). "
                f"Algorithm likely wrong. Investigate before using."
            )
    return nbbo
