"""Sanity tests for Phase G shared helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _phase_g_common as g                                       # noqa: E402

from hft.simulators.adv_cache import get_adv                       # noqa: E402


def test_phase_g_helpers_imports_clean():
    """Module imports without error and exposes expected attrs."""
    assert hasattr(g, "WINDOWS")
    assert hasattr(g, "OOS_DATE")
    assert hasattr(g, "TRAIN_DATES")
    assert hasattr(g, "list_eval_tickers")
    assert hasattr(g, "make_strategies")
    assert hasattr(g, "load_rl_strategy")


def test_window_definitions_cover_rth():
    """5 windows; first starts at 09:30, last ends at 16:00."""
    assert len(g.WINDOWS) == 5
    names = [w[0] for w in g.WINDOWS]
    assert names == ["open", "early_mid", "mid", "late_mid", "close"]
    # First window: 09:30 start
    assert g.WINDOWS[0][1] == 9 * 3600 + 30 * 60
    # Last window: 16:00 end
    assert g.WINDOWS[-1][2] == 16 * 3600
    # Each window is exactly 1 hour
    for name, start, end in g.WINDOWS:
        assert end - start == 3600, f"{name}: expected 1 hour, got {(end-start)/3600}h"


def test_early_mid_window_matches_phase_e_baseline():
    """early_mid window == Phase E.3 OOS baseline (10:00-11:00 ET).
    This is required so the regression sanity check (RL median IS = +1.55)
    is comparable across phases.
    """
    em = g.WINDOWS[1]
    assert em[0] == "early_mid"
    assert em[1] == 10 * 3600
    assert em[2] == 11 * 3600


def test_close_window_covers_last_hour():
    """close window (15:00-16:00) is the gap Lucas identified — must exist."""
    close = g.WINDOWS[-1]
    assert close[0] == "close"
    assert close[1] == 15 * 3600
    assert close[2] == 16 * 3600


def test_train_dates_exclude_oos():
    """TRAIN_DATES must not include the OOS day."""
    assert g.OOS_DATE not in g.TRAIN_DATES
    assert len(g.TRAIN_DATES) == 4


def test_list_eval_tickers_returns_104():
    """Eval pool should be ~100 tickers with full 5-day coverage."""
    tickers = g.list_eval_tickers()
    # Allow some flexibility — should be > 90
    assert len(tickers) >= 90, f"expected ≥ 90 tickers, got {len(tickers)}"
    # Must include core large-caps
    for t in ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "TSLA"]:
        assert t in tickers, f"missing {t}"


def test_get_adv_excludes_oos():
    """ADV computation must exclude the OOS day."""
    adv_with_oos = get_adv("AAPL", exclude_dates=[])
    adv_without_oos = get_adv("AAPL", exclude_dates=[g.OOS_DATE])
    # Should differ (Day 5 contributes to "with OOS" but not "without")
    assert adv_with_oos != adv_without_oos
    # Both should be positive
    assert adv_with_oos > 0 and adv_without_oos > 0


def test_make_strategies_returns_6_with_rl():
    """If RL model exists, make_strategies returns 6; else 5."""
    rl_strat = g.load_rl_strategy()
    strats = g.make_strategies("AAPL", adv=33e6, rl_strat=rl_strat)
    if rl_strat is not None:
        assert len(strats) == 6
        names = [s[0] for s in strats]
        assert g.RL_MODEL_NAME in names
    else:
        assert len(strats) == 5
    # Always have these 5 classical
    names = [s[0] for s in strats]
    for required in ["twap", "ac_ra", "vwap_following", "pov_5pct_cap", "toth"]:
        assert required in names


def test_parent_size_scales_with_adv():
    """Sanity: % of ADV produces reasonable integer share counts."""
    # AAPL ADV ~ 33M; 1% = 330k shares (reasonable for hourly execution)
    aapl_adv = get_adv("AAPL", exclude_dates=[g.OOS_DATE])
    qty_01pct = int(round(aapl_adv * 0.001))
    qty_1pct = int(round(aapl_adv * 0.01))
    qty_5pct = int(round(aapl_adv * 0.05))
    assert qty_01pct > 100, f"0.1% AAPL ADV must be > 100 shares, got {qty_01pct}"
    # Allow small rounding drift across the multiplications.
    assert abs(qty_1pct - 10 * qty_01pct) <= 10
    assert abs(qty_5pct - 50 * qty_01pct) <= 50
