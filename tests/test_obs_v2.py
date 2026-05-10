"""Tests for the 13-dim v2 microstructure observation builder."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from hft.analysis.nbbo_lookup import NBBOLookup, VenueBBOLookup
from hft.data import load_eq_taq
from hft.data.timeparse import add_eq_ns_of_day, filter_rth
from hft.simulators.execution_env import ExecutionEnv
from hft.simulators.obs_v2 import (
    OBS_V2_HIGH,
    OBS_V2_LOW,
    OBS_V2_NAMES,
    TICKER_IDX,
    build_obs_v2,
)
from hft.simulators.vol_profile_baseline import VolProfileBaseline


NS_PER_HOUR = 3600 * 1_000_000_000


@pytest.fixture(scope="module")
def aapl_helpers():
    df = load_eq_taq("AAPL", "20200113")
    df = add_eq_ns_of_day(df)
    df = filter_rth(df, src="Timestamp")
    return {
        "df": df,
        "nbbo_lookup": NBBOLookup(df),
        "venue_bbo_lookup": VenueBBOLookup(df),
        "trades_df": df.filter(pl.col("EventType").is_in(["TRADE", "TRADE NB"]))
            .select(["ns_of_day", "Price", "Quantity"])
            .sort("ns_of_day"),
        "vol_profile": VolProfileBaseline.for_ticker(
            ticker="AAPL", exclude_dates=["20200117"],
        ),
    }


def test_obs_v2_dim_matches_box_bounds():
    """13 dims, low/high/names all the same length."""
    assert len(OBS_V2_LOW) == 13
    assert len(OBS_V2_HIGH) == 13
    assert len(OBS_V2_NAMES) == 13


def test_obs_v2_in_bounds_for_real_anchor(aapl_helpers):
    """Computed obs at a real anchor sits inside the published box."""
    anchor_ns = 10 * NS_PER_HOUR + 30 * 60 * 1_000_000_000  # 10:30 ET
    fake_mids = np.full(3600, 313.0)
    obs = build_obs_v2(
        inventory_left=10_000, total_qty=10_000,
        step_idx=0, n_steps=120, cumulative_filled=0.0,
        anchor_ns=anchor_ns, episode_arrival_mid=313.0,
        episode_mids=fake_mids, step_seconds=30,
        ticker_idx=TICKER_IDX["AAPL"],
        nbbo_lookup=aapl_helpers["nbbo_lookup"],
        venue_bbo_lookup=aapl_helpers["venue_bbo_lookup"],
        trades_df=aapl_helpers["trades_df"],
        vol_profile_baseline=aapl_helpers["vol_profile"],
    )
    assert obs.shape == (13,)
    assert np.isfinite(obs).all(), f"non-finite values: {obs}"
    assert (obs >= OBS_V2_LOW).all(), f"below low: {obs - OBS_V2_LOW}"
    assert (obs <= OBS_V2_HIGH).all(), f"above high: {obs - OBS_V2_HIGH}"


def test_obs_v2_no_lookahead(aapl_helpers):
    """The obs at anchor t must NOT depend on data after t.

    Approach: compute obs at anchor T, then increase the *trades_df* by
    appending a fake huge trade AT t + 1 second. Re-compute obs at T —
    must be identical.
    """
    anchor_ns = 10 * NS_PER_HOUR + 30 * 60 * 1_000_000_000  # 10:30 ET
    fake_mids = np.full(3600, 313.0)
    args = dict(
        inventory_left=10_000, total_qty=10_000,
        step_idx=0, n_steps=120, cumulative_filled=0.0,
        anchor_ns=anchor_ns, episode_arrival_mid=313.0,
        episode_mids=fake_mids, step_seconds=30,
        ticker_idx=TICKER_IDX["AAPL"],
        nbbo_lookup=aapl_helpers["nbbo_lookup"],
        venue_bbo_lookup=aapl_helpers["venue_bbo_lookup"],
        vol_profile_baseline=aapl_helpers["vol_profile"],
    )
    obs_before = build_obs_v2(trades_df=aapl_helpers["trades_df"], **args)
    poisoned = pl.concat([
        aapl_helpers["trades_df"],
        pl.DataFrame({
            "ns_of_day": [anchor_ns + 1_000_000_000],
            "Price": [9999.0],
            "Quantity": [1_000_000],
        }, schema={"ns_of_day": pl.Int64, "Price": pl.Float64, "Quantity": pl.Int64}),
    ]).sort("ns_of_day")
    obs_after = build_obs_v2(trades_df=poisoned, **args)
    np.testing.assert_array_equal(
        obs_before, obs_after,
        err_msg="obs changed after appending future trade — look-ahead leak",
    )


def test_vol_profile_baseline_excludes_target_date():
    """Baseline built with exclude=Day 5 must not load Day 5 data."""
    bp = VolProfileBaseline.for_ticker(
        ticker="AAPL", exclude_dates=["20200117"],
    )
    # 5 days available, exclude 1 → 4 days
    assert bp.n_days == 4
    assert bp.ticker == "AAPL"


def test_vol_profile_cumulative_pct_monotone():
    """cumulative_pct_at must be non-decreasing across the day."""
    bp = VolProfileBaseline.for_ticker(
        ticker="AAPL", exclude_dates=["20200117"],
    )
    samples_ns = np.linspace(
        9 * 3600 * int(1e9) + 30 * 60 * int(1e9),  # 09:30
        16 * 3600 * int(1e9),                       # 16:00
        50, dtype=np.int64,
    )
    pcts = [bp.cumulative_pct_at(int(ns)) for ns in samples_ns]
    assert pcts[0] <= 0.05, f"start should be near 0, got {pcts[0]}"
    assert pcts[-1] >= 0.95, f"end should be near 1, got {pcts[-1]}"
    for a, b in zip(pcts, pcts[1:]):
        assert b >= a - 1e-9, f"non-monotone: {a} → {b}"


def test_executionenv_v2_60min_in_bounds():
    """Full env round-trip: 60-min v2 reset + step, obs always in box."""
    env = ExecutionEnv(
        mode="real_replay",
        ticker="AAPL", date="20200113",
        slice_minutes=60, step_seconds=30, n_steps=120,
        observation_mode="v2",
        max_action_per_step=0.05,
        seed=0,
    )
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    for _ in range(10):
        obs, reward, term, trunc, info = env.step(np.array([0.02], dtype=np.float32))
        assert env.observation_space.contains(obs), f"obs out of bounds: {obs}"
        assert np.isfinite(reward)
        if term or trunc:
            break


def test_executionenv_v2_rejects_synthetic_mode():
    """v2 obs requires real TAQ data; synthetic mode must be rejected loudly."""
    with pytest.raises(ValueError, match="observation_mode='v2' requires"):
        ExecutionEnv(mode="synthetic", observation_mode="v2")


def test_executionenv_ticker_pool_requires_real_replay():
    """ticker_pool must error if mode != real_replay."""
    with pytest.raises(ValueError, match="ticker_pool requires"):
        ExecutionEnv(mode="synthetic", ticker_pool=["AAPL", "AMZN"])


def test_executionenv_multi_ticker_pool_samples_diversity():
    """ticker_pool reset() must reach multiple tickers across many resets."""
    env = ExecutionEnv(
        mode="real_replay",
        ticker="AAPL",
        ticker_pool=["AAPL", "AMZN", "TSLA"],
        date_pool=["20200113", "20200114"],
        slice_minutes=60, step_seconds=30, n_steps=120,
        observation_mode="v2",
        seed=42,
    )
    seen_tickers = set()
    for i in range(60):
        env.reset(seed=42 + i)
        seen_tickers.add(env._current_ticker)
    assert len(seen_tickers) >= 2, (
        f"expected ≥ 2 unique tickers, got {seen_tickers}"
    )


# ---------------------------------------------------------------------------
# Phase E — v3 (ticker-agnostic obs) + spread-cost fills
# ---------------------------------------------------------------------------

def test_obs_v3_dim_and_bounds_consistent():
    """v3 has 13 dims; low/high/names align."""
    from hft.simulators.obs_v2 import OBS_V3_LOW, OBS_V3_HIGH, OBS_V3_NAMES
    assert len(OBS_V3_LOW) == 13
    assert len(OBS_V3_HIGH) == 13
    assert len(OBS_V3_NAMES) == 13
    assert OBS_V3_NAMES[12] == "log_adv_norm", "dim 12 must be log_adv_norm"
    assert "ticker_idx" not in OBS_V3_NAMES, "ticker_idx must be dropped in v3"


def test_obs_v3_in_box_for_unseen_ticker():
    """v3 obs computes valid features on tickers outside the Phase D pool."""
    env = ExecutionEnv(
        mode="real_replay",
        ticker="MSFT", date="20200117",   # MSFT was NOT in Phase D's 5-ticker pool
        slice_minutes=60, step_seconds=30, n_steps=120,
        observation_mode="v3",
        max_action_per_step=0.05,
        seed=0,
    )
    obs, info = env.reset(seed=0, options={"force_anchor": {
        "ticker": "MSFT", "date": "20200117",
        "start_ns": int(10 * 3600 * 1e9),
    }})
    assert obs.shape == (13,)
    assert env.observation_space.contains(obs), f"obs out of bounds: {obs}"
    # log_adv_norm should be in [-2, 2]
    assert -2.0 <= obs[12] <= 2.0


def test_fill_at_spread_lower_reward_than_mid():
    """Same episode, same action sequence: fill_at_spread=True must give
    strictly lower (or equal) reward than mid-fill (half-spread cost).
    """
    opts = {"force_anchor": {
        "ticker": "AAPL", "date": "20200113",
        "start_ns": int(10 * 3600 * 1e9),
    }}

    def run(fill_at_spread: bool) -> float:
        env = ExecutionEnv(
            mode="real_replay", ticker="AAPL", date="20200113",
            slice_minutes=5, step_seconds=5, n_steps=60,
            fill_at_spread=fill_at_spread,
            seed=0,
        )
        env.reset(seed=0, options=opts)
        total = 0.0
        for _ in range(60):
            _, r, term, trunc, _ = env.step(np.array([0.05], dtype=np.float32))
            total += r
            if term or trunc:
                break
        return total

    r_mid = run(False)
    r_spread = run(True)
    assert r_spread < r_mid, (
        f"spread fills must cost half-spread vs mid; got mid={r_mid:.4f}, "
        f"spread={r_spread:.4f}"
    )


def test_fill_at_spread_rejected_on_synthetic_mode():
    """fill_at_spread requires real_replay (synthetic has no NBBO)."""
    with pytest.raises(ValueError, match="fill_at_spread requires"):
        ExecutionEnv(mode="synthetic", fill_at_spread=True)


def test_adv_cache_returns_finite_for_known_tickers():
    """ADV cache produces positive finite ADV for tickers in the cache."""
    from hft.simulators.adv_cache import get_adv
    for t in ["AAPL", "MSFT", "TSLA"]:
        adv = get_adv(t, exclude_dates=["20200117"])
        assert adv > 0 and np.isfinite(adv), f"ADV bad for {t}: {adv}"
