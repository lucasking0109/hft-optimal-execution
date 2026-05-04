"""Unit tests for ACMLStrategy (Phase B.3 Step 4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from hft.strategies.ac_ml import ACMLStrategy
from hft.strategies.base import ParentOrder

ROOT = Path(__file__).resolve().parents[1]
NS_PER_HOUR = 3600 * 1_000_000_000
MODEL_PATH = ROOT / "rl" / "checkpoints" / "eta_ml_v0" / "model.json"


@pytest.fixture(scope="module")
def trained_model_required():
    if not MODEL_PATH.exists():
        pytest.skip(
            f"Model not yet trained ({MODEL_PATH}). "
            f"Run scripts/train_eta_ml_model.py to enable these tests."
        )


def test_ac_ml_validates_model_path():
    """Missing model → explicit ValueError (NO Silent Fallback)."""
    with pytest.raises(ValueError, match="model not found"):
        ACMLStrategy(
            model_path="/nonexistent/path/model.json",
            adv_shares=1e7,
            sigma_bps_per_sqrt_sec=1.0,
        )


def test_ac_ml_validates_adv(trained_model_required):
    with pytest.raises(ValueError, match="adv_shares"):
        ACMLStrategy(
            model_path=str(MODEL_PATH),
            adv_shares=0,
            sigma_bps_per_sqrt_sec=1.0,
        )


def test_ac_ml_predicted_eta_within_range(trained_model_required):
    """Predicted η must clamp to [eta_floor, eta_ceiling] range."""
    from hft.data import load_eq_taq
    strat = ACMLStrategy(
        model_path=str(MODEL_PATH),
        adv_shares=33_809_763.0,   # AAPL ADV
        sigma_bps_per_sqrt_sec=1.0,
        eta_floor=0.5,
        eta_ceiling=50.0,
    )
    df = load_eq_taq("AAPL", "20200113")
    eta = strat.predict_eta(
        ticker="AAPL", df=df,
        anchor_ns=int(10 * NS_PER_HOUR),
        parent_qty=10_000,
    )
    assert 0.5 <= eta <= 50.0, f"eta {eta} outside literature range"


def test_ac_ml_requires_market_df(trained_model_required):
    """Missing market_df in context → explicit ValueError."""
    strat = ACMLStrategy(
        model_path=str(MODEL_PATH),
        adv_shares=1e7,
        sigma_bps_per_sqrt_sec=1.0,
    )
    parent = ParentOrder(
        ticker="AAPL", date="20200113", side="sell",
        quantity=10_000,
        start_ns=int(10 * NS_PER_HOUR),
        end_ns=int(11 * NS_PER_HOUR),
    )
    with pytest.raises(ValueError, match="market_df"):
        strat.schedule(parent, market_context={})


def test_ac_ml_schedule_completes(trained_model_required):
    """Schedule should sum to parent.quantity and fit in window."""
    from hft.data import load_eq_taq
    strat = ACMLStrategy(
        model_path=str(MODEL_PATH),
        adv_shares=33_809_763.0,
        sigma_bps_per_sqrt_sec=1.0,
        lambda_risk=1e-3,
        num_slices=30,
    )
    df = load_eq_taq("AAPL", "20200113")
    parent = ParentOrder(
        ticker="AAPL", date="20200113", side="sell",
        quantity=10_000,
        start_ns=int(10 * NS_PER_HOUR),
        end_ns=int(11 * NS_PER_HOUR),
    )
    children = strat.schedule(parent, market_context={"market_df": df})
    assert sum(c.quantity for c in children) == 10_000
    assert all(parent.start_ns <= c.timestamp_ns < parent.end_ns for c in children)
    assert hasattr(strat, "last_predicted_eta")
