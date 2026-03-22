"""Tests for Phase 3 impact estimation and Almgren-Chriss strategy."""

from __future__ import annotations

import math

import pytest

from hft.analysis.impact import (
    GAMMA_PRIOR_BPS_PER_PCT_ADV,
    average_daily_volume,
    check_gamma_against_prior,
    estimate_eta,
)
from hft.data import load_eq_daily_ohlc, load_eq_taq
from hft.strategies.almgren_chriss import (
    AlmgrenChrissStrategy,
    estimate_intraday_sigma_bps_per_sqrt_sec,
)
from hft.strategies.base import ParentOrder

NS_PER_HOUR = 3600 * 1_000_000_000


# ---------------------------------------------------------------------------
# Eta estimation on real AAPL data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def aapl_day():
    return load_eq_taq("AAPL", "20200113")


@pytest.fixture(scope="module")
def aapl_adv():
    daily = load_eq_daily_ohlc("20200113")
    return average_daily_volume(daily, "AAPL", lookback_days=1)


def test_estimate_eta_runs_on_aapl(aapl_day, aapl_adv):
    res = estimate_eta(aapl_day, adv_shares=aapl_adv, percentile_threshold=0.90)
    # We expect the slope to be in a sane bps/%ADV range
    assert res.n_events >= 30
    assert -50 < res.eta_bps_per_pct_adv < 50, (
        f"η={res.eta_bps_per_pct_adv} unreasonable; check estimation code"
    )
    # We don't fail the test if it's out_of_range; we want to *see* what real
    # data tells us. The flag is for the report, not test gating.


def test_check_gamma_against_prior_runs(aapl_day, aapl_adv):
    chk = check_gamma_against_prior(
        aapl_day, ticker="AAPL", adv_shares=aapl_adv,
        long_horizon_seconds=600, percentile_threshold=0.95,
    )
    assert chk.prior_bps_per_pct_adv == GAMMA_PRIOR_BPS_PER_PCT_ADV["AAPL"]
    # Ratio should be a number (or NaN if too few events)
    assert isinstance(chk.note, str)


def test_estimate_eta_rejects_too_few_events(aapl_day, aapl_adv):
    # Set absurd percentile so we get few events
    with pytest.raises(ValueError):
        estimate_eta(
            aapl_day, adv_shares=aapl_adv,
            percentile_threshold=0.999999, min_events=10000,
        )


# ---------------------------------------------------------------------------
# Almgren-Chriss strategy unit tests
# ---------------------------------------------------------------------------

def _parent(qty=10_000):
    return ParentOrder(
        ticker="AAPL", date="20200113", side="sell",
        quantity=qty, start_ns=int(10 * NS_PER_HOUR), end_ns=int(11 * NS_PER_HOUR),
    )


def test_ac_risk_neutral_is_linear():
    """At λ=0 AC must equal TWAP."""
    strat = AlmgrenChrissStrategy(
        num_slices=10, eta_bps_per_pct_adv=10.0,
        gamma_bps_per_pct_adv=5.0, sigma_bps_per_sqrt_sec=1.0,
        lambda_risk=0.0, adv_shares=1_000_000,
    )
    children = strat.schedule(_parent(qty=1000), market_context={})
    qtys = [c.quantity for c in children]
    assert all(q == 100 for q in qtys), f"risk-neutral AC should give 100 each, got {qtys}"


def test_ac_risk_averse_front_loaded():
    """Risk-averse AC should sell more in the early slices than the late."""
    strat = AlmgrenChrissStrategy(
        num_slices=10, eta_bps_per_pct_adv=10.0,
        gamma_bps_per_pct_adv=5.0,
        sigma_bps_per_sqrt_sec=5.0, lambda_risk=1e-3,
        adv_shares=1_000_000,
    )
    children = strat.schedule(_parent(qty=10_000), market_context={})
    qtys = [c.quantity for c in children]
    assert sum(qtys) == 10_000
    # First slice quantity should exceed last slice
    assert qtys[0] > qtys[-1], f"expected front-load, got first={qtys[0]} last={qtys[-1]}"


def test_ac_validates_inputs():
    with pytest.raises(ValueError):
        AlmgrenChrissStrategy(
            num_slices=10, eta_bps_per_pct_adv=-1, gamma_bps_per_pct_adv=5,
            sigma_bps_per_sqrt_sec=1, lambda_risk=0, adv_shares=1e6,
        )
    with pytest.raises(ValueError):
        AlmgrenChrissStrategy(
            num_slices=10, eta_bps_per_pct_adv=10, gamma_bps_per_pct_adv=5,
            sigma_bps_per_sqrt_sec=1, lambda_risk=0, adv_shares=0,
        )


def test_ac_total_quantity_preserved_with_rounding():
    strat = AlmgrenChrissStrategy(
        num_slices=37, eta_bps_per_pct_adv=10, gamma_bps_per_pct_adv=5,
        sigma_bps_per_sqrt_sec=5, lambda_risk=1e-3, adv_shares=1e6,
    )
    children = strat.schedule(_parent(qty=12345), market_context={})
    assert sum(c.quantity for c in children) == 12345


def test_estimate_sigma(aapl_day):
    sigma = estimate_intraday_sigma_bps_per_sqrt_sec(aapl_day, sample_seconds=60)
    # AAPL intraday typical σ on a normal day: ~ 1–10 bps per √sec
    assert 0.1 < sigma < 50, f"σ={sigma} bps/√sec looks wrong"
