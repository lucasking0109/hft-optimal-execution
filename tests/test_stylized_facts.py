"""Tests for stylized facts module — toy data + real AAPL + abides."""

from __future__ import annotations

import numpy as np
import pytest

from hft.simulators.stylized_facts import (
    StylizedFacts,
    compute_stylized_facts,
    extract_abides_series,
    extract_real_taq_series,
    facts_distance,
    total_calibration_distance,
)


# ---------------------------------------------------------------------------
# Toy data tests — no external deps
# ---------------------------------------------------------------------------

def test_compute_stylized_facts_gaussian_returns():
    """Gaussian returns: kurtosis ≈ 0, no fat tail."""
    rng = np.random.default_rng(0)
    log_mid = np.cumsum(rng.normal(0, 1e-4, 5000))
    mids = 100 * np.exp(log_mid)
    spread = rng.uniform(0.5, 2.0, 5000)
    trades = rng.integers(10, 200, 1000)
    facts = compute_stylized_facts(
        mid_prices=mids, spread_bps=spread, trade_sizes=trades,
    )
    # Gaussian → excess kurtosis near 0
    assert -1.5 < facts.excess_kurtosis < 1.5
    # Hill on Gaussian tail estimates around 5-30 (varies, just sanity check)
    assert facts.hill_tail_index > 1.0
    # Vol autocorr length 100
    assert len(facts.vol_autocorr) == 100
    assert facts.n_returns == 4999


def test_compute_stylized_facts_fat_tail_returns():
    """Student-t (df=3) returns: excess kurtosis should be HIGH."""
    rng = np.random.default_rng(42)
    raw_returns = rng.standard_t(df=3, size=5000) * 1e-4
    log_mid = np.cumsum(raw_returns)
    mids = 100 * np.exp(log_mid)
    spread = rng.uniform(0.5, 2.0, 5000)
    trades = rng.integers(10, 200, 1000)
    facts = compute_stylized_facts(
        mid_prices=mids, spread_bps=spread, trade_sizes=trades,
    )
    # Student-t df=3 has theoretical kurtosis = ∞ (very large empirically)
    assert facts.excess_kurtosis > 2.0


def test_compute_stylized_facts_rejects_short_input():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="mid_prices"):
        compute_stylized_facts(
            mid_prices=rng.normal(100, 1, 50),
            spread_bps=rng.uniform(0.5, 2.0, 100),
            trade_sizes=rng.integers(10, 200, 100),
        )


def test_facts_distance_self_is_zero():
    """Distance between identical facts should be ~0."""
    rng = np.random.default_rng(0)
    log_mid = np.cumsum(rng.normal(0, 1e-4, 5000))
    mids = 100 * np.exp(log_mid)
    facts = compute_stylized_facts(
        mid_prices=mids,
        spread_bps=rng.uniform(0.5, 2.0, 5000),
        trade_sizes=rng.integers(10, 200, 1000),
    )
    d = facts_distance(facts, facts)
    assert abs(d["kurtosis_ratio"] - 1.0) < 1e-9
    assert abs(d["hill_ratio"] - 1.0) < 1e-9
    assert d["spread_ks_dist"] < 1e-9
    assert d["trade_size_ks_dist"] < 1e-9
    assert d["autocorr_l2"] < 1e-9


def test_facts_distance_different():
    """Distance between different distributions should be > 0."""
    rng = np.random.default_rng(0)
    facts_a = compute_stylized_facts(
        mid_prices=100 * np.exp(np.cumsum(rng.normal(0, 1e-4, 3000))),
        spread_bps=rng.uniform(0.5, 1.5, 3000),
        trade_sizes=rng.integers(10, 100, 800),
    )
    rng2 = np.random.default_rng(1)
    facts_b = compute_stylized_facts(
        mid_prices=100 * np.exp(np.cumsum(rng2.normal(0, 5e-4, 3000))),  # 5x volatility
        spread_bps=rng2.uniform(5.0, 15.0, 3000),                         # much wider spread
        trade_sizes=rng2.integers(100, 500, 800),                         # bigger trades
    )
    d = facts_distance(facts_a, facts_b)
    # Spread distributions clearly different → KS > 0.5
    assert d["spread_ks_dist"] > 0.5
    # Trade size distributions clearly different
    assert d["trade_size_ks_dist"] > 0.5
    # Total distance non-trivial
    total = total_calibration_distance(d)
    assert total > 0.5


def test_total_calibration_distance_handles_nan():
    """NaN ratios shouldn't crash; should give finite (penalty) value."""
    fake_distances = {
        "kurtosis_ratio": float("nan"),
        "hill_ratio": -0.5,
        "spread_ks_dist": 0.3,
        "trade_size_ks_dist": 0.2,
        "autocorr_l2": 0.5,
    }
    d = total_calibration_distance(fake_distances)
    assert np.isfinite(d)
    assert d > 0


# ---------------------------------------------------------------------------
# Real AAPL integration test
# ---------------------------------------------------------------------------

def test_extract_real_taq_series_aapl():
    from hft.data import load_eq_taq

    df = load_eq_taq("AAPL", "20200113")
    mids, spread, trades = extract_real_taq_series(df, sample_seconds=1, rth_only=True)
    assert len(mids) > 1000
    assert len(spread) > 1000
    assert len(trades) > 1000
    # Real AAPL price around $313 in Jan 2020
    assert 200 < mids.mean() < 400
    # Real AAPL spread typically < 5 bps
    assert np.median(spread) < 5.0


def test_real_aapl_stylized_facts():
    from hft.data import load_eq_taq

    df = load_eq_taq("AAPL", "20200113")
    mids, spread, trades = extract_real_taq_series(df, sample_seconds=1)
    facts = compute_stylized_facts(
        mid_prices=mids, spread_bps=spread, trade_sizes=trades,
        metadata={"source": "real_aapl_20200113"},
    )
    # Real AAPL high-freq returns: positive excess kurtosis (fat tails)
    # Allow wide range — the test is just that fat tail is present
    assert facts.excess_kurtosis > 1.0
    # Hill estimator → tail alpha typically 2-6 for equity tick
    assert 1.0 < facts.hill_tail_index < 50.0
    # Vol autocorr lag 1 should be positive (volatility clustering)
    assert facts.vol_autocorr[0] > 0


# ---------------------------------------------------------------------------
# abides extraction integration test
# ---------------------------------------------------------------------------

def test_extract_abides_series_smoke():
    from hft.data.abides_loader import load_abides_run

    data = load_abides_run()
    mids, spread, trades = extract_abides_series(data, sample_seconds=1)
    assert len(mids) > 50
    assert len(spread) > 50
    assert len(trades) > 50
    # abides uses rmsc03 default fundamental ~ $1000
    assert 800 < mids.mean() < 1200


def test_abides_facts_vs_real_distance():
    """Sanity: abides vs real AAPL — distances should be > 0 (they're different)."""
    from hft.data import load_eq_taq
    from hft.data.abides_loader import load_abides_run

    real_df = load_eq_taq("AAPL", "20200113")
    real_mids, real_spread, real_trades = extract_real_taq_series(real_df, sample_seconds=1)
    real_facts = compute_stylized_facts(
        mid_prices=real_mids, spread_bps=real_spread, trade_sizes=real_trades,
    )

    abides_data = load_abides_run()
    a_mids, a_spread, a_trades = extract_abides_series(abides_data, sample_seconds=1)
    abides_facts = compute_stylized_facts(
        mid_prices=a_mids, spread_bps=a_spread, trade_sizes=a_trades,
    )

    d = facts_distance(real_facts, abides_facts)
    # Spread distribution distance should be HIGH because abides spread ~86 bps vs AAPL <1 bps
    assert d["spread_ks_dist"] > 0.5, "abides vs real spread should differ a lot pre-calibration"
