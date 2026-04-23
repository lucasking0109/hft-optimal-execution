"""Sanity tests for the AgGrid DOM ladder + Time & Sales builders."""

from __future__ import annotations

import polars as pl
import pytest

from hft.viz.abides_lob_grid import (
    dom_grid_options,
    make_dom_ladder_df,
    make_trade_tape_df,
    trade_tape_grid_options,
)


# ---------------------------------------------------------------------------
# Toy snapshot for offline checks (no abides dependency)
# ---------------------------------------------------------------------------

def _toy_snapshot() -> list[dict]:
    return [
        {"price_cents": 100020, "price_dollars": 1000.20, "side": "ASK", "qty": 50},
        {"price_cents": 100018, "price_dollars": 1000.18, "side": "ASK", "qty": 30},
        {"price_cents": 100012, "price_dollars": 1000.12, "side": "ASK", "qty": 100},  # best ask
        {"price_cents": 100011, "price_dollars": 1000.11, "side": "BID", "qty": 80},   # best bid
        {"price_cents": 100008, "price_dollars": 1000.08, "side": "BID", "qty": 60},
        {"price_cents": 100005, "price_dollars": 1000.05, "side": "BID", "qty": 40},
        {"price_cents":  99950, "price_dollars":  999.50, "side": "BID", "qty": 100000},  # boundary
    ]


def test_make_dom_ladder_df_basic():
    df, meta = make_dom_ladder_df(_toy_snapshot(), n_levels=3, filter_boundary_qty=5000)
    assert meta["best_bid_cents"] == 100011
    assert meta["best_ask_cents"] == 100012
    # Boundary order qty=100000 should have been filtered out
    # Ladder spans best_ask + 3 to best_bid - 3 = 100015 down to 100008
    prices_in_ladder = [int(round(float(p) * 100)) for p in df["price"].astype(float)]
    assert max(prices_in_ladder) == 100015
    assert min(prices_in_ladder) == 100008


def test_make_dom_ladder_df_marks_best_bid_and_ask():
    df, meta = make_dom_ladder_df(_toy_snapshot(), n_levels=3, filter_boundary_qty=5000)
    bb_rows = df[df["_is_best_bid"]]
    ba_rows = df[df["_is_best_ask"]]
    assert len(bb_rows) == 1
    assert len(ba_rows) == 1
    assert float(bb_rows.iloc[0]["price"]) == pytest.approx(1000.11)
    assert float(ba_rows.iloc[0]["price"]) == pytest.approx(1000.12)


def test_make_dom_ladder_df_empty_snapshot():
    df, meta = make_dom_ladder_df([], n_levels=5)
    assert df.empty
    assert meta == {}


def test_make_dom_ladder_df_show_filtered_changes_count():
    """show_filtered=True keeps boundary orders → meta.max_qty should be huge."""
    _, meta_filt = make_dom_ladder_df(_toy_snapshot(), n_levels=3, show_filtered=False)
    _, meta_unfilt = make_dom_ladder_df(_toy_snapshot(), n_levels=3, show_filtered=True)
    # Without filter, max_qty includes the 100,000-share boundary order
    assert meta_unfilt["max_qty"] >= 100_000
    assert meta_filt["max_qty"] < 1000


def test_dom_grid_options_returns_dict():
    df, meta = make_dom_ladder_df(_toy_snapshot(), n_levels=3, filter_boundary_qty=5000)
    opts = dom_grid_options(df, meta)
    assert isinstance(opts, dict)
    assert "columnDefs" in opts


def test_make_trade_tape_df_basic():
    import datetime as dt
    df = pl.DataFrame({
        "time": [
            dt.datetime(2020, 1, 13, 9, 30, 0, 100000),
            dt.datetime(2020, 1, 13, 9, 30, 0, 200000),
            dt.datetime(2020, 1, 13, 9, 30, 0, 300000),
        ],
        "quantity": [10, 20, 30],
        "price_dollars": [100.0, 100.05, 100.10],
        "side": ["buy", "sell", "buy"],
    })
    tape = make_trade_tape_df(df, n_recent=10)
    assert len(tape) == 3
    # Newest first
    assert tape.iloc[0]["qty"] == 30
    assert tape.iloc[0]["side"] == "BUY"
    assert tape.iloc[2]["qty"] == 10


def test_trade_tape_grid_options():
    import datetime as dt
    df = pl.DataFrame({
        "time": [dt.datetime(2020, 1, 13, 9, 30, 0)],
        "quantity": [10],
        "price_dollars": [100.0],
        "side": ["buy"],
    })
    tape = make_trade_tape_df(df, n_recent=1)
    opts = trade_tape_grid_options(tape)
    assert "columnDefs" in opts
