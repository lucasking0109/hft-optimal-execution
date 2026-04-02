"""Phase 4 venue analysis tests."""

from __future__ import annotations

import polars as pl
import pytest

from hft.analysis.sor import (
    SORWeights,
    compute_composite_score,
    naive_volume_allocation,
    sor_score_allocation,
)
from hft.analysis.venue_metrics import (
    NON_ROUTABLE_VENUES,
    compute_all_venue_metrics,
    compute_venue_adverse_selection,
    compute_venue_depth,
    compute_venue_nbbo_share,
    compute_venue_nbbo_share_hourly,
    compute_venue_volume_share,
)
from hft.data import load_eq_taq
from hft.viz.venue_routing import (
    plot_adverse_selection_bar,
    plot_composite_score_bar,
    plot_venue_score_heatmap,
)


@pytest.fixture(scope="module")
def aapl_day():
    return load_eq_taq("AAPL", "20200113")


def test_volume_share_sums_to_100(aapl_day):
    out = compute_venue_volume_share(aapl_day)
    assert abs(float(out["volume_share_pct"].sum()) - 100.0) < 0.5


def test_nbbo_share_each_side_sums_to_100(aapl_day):
    out = compute_venue_nbbo_share(aapl_day)
    assert abs(float(out["nbbo_share_bid_pct"].sum()) - 100.0) < 1.0
    assert abs(float(out["nbbo_share_ask_pct"].sum()) - 100.0) < 1.0


def test_depth_returns_positive_values(aapl_day):
    out = compute_venue_depth(aapl_day)
    assert (out["avg_depth_at_best"] > 0).all()


def test_adverse_selection_runs(aapl_day):
    out = compute_venue_adverse_selection(aapl_day, horizon_seconds=60)
    assert "adverse_selection_bps" in out.columns
    assert len(out) > 1   # at least 2 venues


def test_compute_all_returns_routable_flag(aapl_day):
    out = compute_all_venue_metrics(aapl_day, horizon_seconds=60)
    assert "routable" in out.columns
    finra = out.filter(pl.col("Exchange") == "FINRA")
    if not finra.is_empty():
        assert not bool(finra["routable"][0])


def test_hourly_share_returns_grid(aapl_day):
    out = compute_venue_nbbo_share_hourly(aapl_day)
    assert "hour_et" in out.columns
    assert "share_pct" in out.columns
    assert len(out) > 0


def test_composite_score_routable_only(aapl_day):
    metrics = compute_all_venue_metrics(aapl_day)
    scored = compute_composite_score(metrics)
    assert "composite_score" in scored.columns
    finra = scored.filter(pl.col("Exchange") == "FINRA")
    if not finra.is_empty():
        assert float(finra["composite_score"][0]) == 0.0


def test_naive_allocation_sums_to_one(aapl_day):
    metrics = compute_all_venue_metrics(aapl_day)
    alloc = naive_volume_allocation(metrics, top_k=5)
    assert len(alloc) <= 5
    assert abs(sum(alloc.values()) - 1.0) < 1e-6


def test_sor_allocation_sums_to_one(aapl_day):
    metrics = compute_all_venue_metrics(aapl_day)
    scored = compute_composite_score(metrics)
    alloc = sor_score_allocation(scored, top_k=5)
    assert len(alloc) <= 5
    assert abs(sum(alloc.values()) - 1.0) < 1e-6


def test_visualizations_return_figures(aapl_day):
    metrics = compute_all_venue_metrics(aapl_day)
    scored = compute_composite_score(metrics)
    hourly = compute_venue_nbbo_share_hourly(aapl_day)

    f1 = plot_venue_score_heatmap(hourly, "AAPL", "20200113")
    f2 = plot_adverse_selection_bar(metrics, "AAPL", "20200113")
    f3 = plot_composite_score_bar(scored, "AAPL", "20200113")
    assert f1 is not None and f2 is not None and f3 is not None
