"""Phase 1 viz module sanity tests.

Each test confirms the module produces a valid plotly Figure on AAPL or NQ
without crashing. We don't pixel-compare; just structural integrity.
"""

from __future__ import annotations

import plotly.graph_objects as go
import pytest

from hft.data import load_eq_taq
from hft.data.orderbook import load_fut_taq
from hft.data.timeparse import (
    add_eq_minute_bucket,
    add_eq_ns_of_day,
    add_fut_ns_of_day,
    filter_rth,
)
from hft.viz.futures_charts import plot_aggressor_flow
from hft.viz.price_charts import plot_price_with_band, plot_trade_size_distribution
from hft.viz.spread import plot_spread
from hft.viz.venue_heatmap import plot_venue_heatmap, plot_venue_summary_bar
from hft.viz.volume_profile import plot_intraday_volume


@pytest.fixture(scope="module")
def aapl_day():
    return load_eq_taq("AAPL", "20200113")


@pytest.fixture(scope="module")
def nq_day():
    return load_fut_taq("NQH0", "20200113")


def test_timeparse_eq_ns_of_day(aapl_day):
    sliced = aapl_day.head(100)
    out = add_eq_ns_of_day(sliced)
    assert "ns_of_day" in out.columns
    # 04:00:00.062779093 → ns
    expected = (4 * 3600) * 1_000_000_000 + 62779093
    assert out["ns_of_day"][0] == expected


def test_timeparse_eq_minute_bucket(aapl_day):
    out = add_eq_minute_bucket(aapl_day.head(50), bin_minutes=5)
    assert "bucket_min" in out.columns
    # 04:00 = minute 240 → bucket 240/5 = 48
    assert out["bucket_min"][0] == 48


def test_timeparse_filter_rth(aapl_day):
    rth = filter_rth(aapl_day)
    assert len(rth) > 0
    assert len(rth) < len(aapl_day)  # should remove pre-/post-market


def test_timeparse_fut_ns_of_day(nq_day):
    out = add_fut_ns_of_day(nq_day.head(20))
    assert "ns_of_day" in out.columns
    assert (out["ns_of_day"] >= 0).all()


def test_price_chart_returns_figure(aapl_day):
    fig = plot_price_with_band(aapl_day, "AAPL", "20200113", sample_every=500)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_volume_profile_returns_figure(aapl_day):
    fig = plot_intraday_volume(aapl_day, "AAPL", "20200113", bin_minutes=5)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    assert fig.data[0].type == "bar"


def test_spread_chart_returns_figure(aapl_day):
    fig = plot_spread(aapl_day, "AAPL", "20200113", sample_every=500)
    assert isinstance(fig, go.Figure)


def test_venue_heatmap_returns_figure(aapl_day):
    fig = plot_venue_heatmap(aapl_day, "AAPL", "20200113", bin_minutes=15)
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "heatmap"


def test_venue_summary_bar_returns_figure(aapl_day):
    fig = plot_venue_summary_bar(aapl_day, "AAPL", "20200113")
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "bar"


def test_aggressor_flow_returns_figure(nq_day):
    fig = plot_aggressor_flow(nq_day, "NQH0", "20200113", sample_every=500)
    assert isinstance(fig, go.Figure)


def test_trade_size_distribution_returns_figure(aapl_day):
    fig = plot_trade_size_distribution(aapl_day, "AAPL", "20200113")
    assert isinstance(fig, go.Figure)
