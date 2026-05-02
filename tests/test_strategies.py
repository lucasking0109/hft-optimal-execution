"""Sanity tests for execution strategies."""

from __future__ import annotations

import polars as pl
import pytest

from hft.strategies.base import ChildOrder, ParentOrder
from hft.strategies.pov import POVStrategy
from hft.strategies.twap import TWAPStrategy
from hft.strategies.vwap_following import VWAPFollowingStrategy

NS_PER_HOUR = 3600 * 1_000_000_000


def _parent(qty: int = 600) -> ParentOrder:
    return ParentOrder(
        ticker="AAPL", date="20200113", side="sell",
        quantity=qty,
        start_ns=int(9.5 * NS_PER_HOUR),
        end_ns=int(10.5 * NS_PER_HOUR),
    )


def test_twap_evenly_distributes():
    strat = TWAPStrategy(num_slices=6)
    children = strat.schedule(_parent(qty=600), market_context={})
    assert len(children) == 6
    qtys = [c.quantity for c in children]
    assert all(q == 100 for q in qtys), f"TWAP should give equal qtys, got {qtys}"
    assert sum(qtys) == 600


def test_twap_handles_remainder():
    strat = TWAPStrategy(num_slices=7)
    children = strat.schedule(_parent(qty=100), market_context={})
    # 100 / 7 = 14 r 2 → first two children get 15, rest 14
    qtys = [c.quantity for c in children]
    assert sum(qtys) == 100
    assert qtys[0] == 15 and qtys[-1] == 14


def test_twap_timestamps_monotonic():
    strat = TWAPStrategy(num_slices=10)
    children = strat.schedule(_parent(), market_context={})
    timestamps = [c.timestamp_ns for c in children]
    assert timestamps == sorted(timestamps)


def test_twap_validates_num_slices():
    with pytest.raises(ValueError):
        TWAPStrategy(num_slices=0)


def test_parent_order_validates_quantity():
    with pytest.raises(ValueError):
        ParentOrder(ticker="X", date="20200113", side="sell",
                    quantity=0, start_ns=0, end_ns=1000)


def test_parent_order_validates_window():
    with pytest.raises(ValueError):
        ParentOrder(ticker="X", date="20200113", side="sell",
                    quantity=100, start_ns=1000, end_ns=500)


def test_vwap_following_requires_profile():
    strat = VWAPFollowingStrategy()
    with pytest.raises(ValueError, match="volume_profile"):
        strat.schedule(_parent(), market_context={})


def test_vwap_following_uses_profile():
    """If profile is U-shape (heavy at edges), schedule reflects that."""
    profile = pl.DataFrame({
        # bin_minutes=5, so bucket 114 = 09:30, 126 = 10:30, 138 = 11:30
        "bucket_min": [114, 120, 126],  # 09:30, 10:00, 10:30 (5-min buckets)
        "volume": [1000, 100, 1000],   # heavy at edges, light middle
    })
    strat = VWAPFollowingStrategy()
    parent = _parent(qty=2100)  # divisible by 1000+100+1000=2100
    children = strat.schedule(parent, market_context={
        "volume_profile": profile, "bin_minutes": 5,
    })
    assert sum(c.quantity for c in children) == 2100
    # Heaviest bucket(s) should get the most quantity
    qtys = [c.quantity for c in children]
    # Inner bucket should be smallest
    assert min(qtys) < max(qtys)


# ---------------------------------------------------------------------------
# POV strategy tests
# ---------------------------------------------------------------------------

def _profile_simple() -> pl.DataFrame:
    """5-min buckets at 09:30/10:00/10:30 with U-shape volumes."""
    return pl.DataFrame({
        "bucket_min": [114, 120, 126],
        "volume": [1000, 100, 1000],
    })


def test_pov_validates_cap():
    with pytest.raises(ValueError, match="cap_pov"):
        POVStrategy(cap_pov=0.0)
    with pytest.raises(ValueError, match="cap_pov"):
        POVStrategy(cap_pov=1.5)


def test_pov_validates_target():
    with pytest.raises(ValueError, match="target_pov"):
        POVStrategy(target_pov=2.0)


def test_pov_auto_target_completes():
    """Auto target_pov should complete the parent quantity (force_completion)."""
    strat = POVStrategy(target_pov=None, cap_pov=0.5, force_completion=True)
    parent = _parent(qty=210)  # 10% of 2100 total volume → target_pov=0.10
    children = strat.schedule(parent, market_context={
        "volume_profile": _profile_simple(), "bin_minutes": 5,
    })
    assert sum(c.quantity for c in children) == 210


def test_pov_cap_under_fills_then_completes():
    """If cap < auto-target, cap binds and residual is water-filled.

    Residual is distributed proportional to bucket volume (post-audit fix);
    every child ends with the same effective POV instead of the residual
    being concentrated in the final child.
    """
    strat = POVStrategy(target_pov=None, cap_pov=0.05, force_completion=True)
    # Auto target would need 0.382 (parent 420 / 1100 eligible vol),
    # but cap=0.05 binds.
    parent = _parent(qty=420)
    children = strat.schedule(parent, market_context={
        "volume_profile": _profile_simple(), "bin_minutes": 5,
    })
    # Sum still equals parent qty due to force_completion water-fill.
    assert sum(c.quantity for c in children) == 420
    # Eligibility: parent ends at 10:30 (=bucket 126 start, NOT inclusive),
    # so only buckets 114 (vol 1000) and 120 (vol 100) are eligible.
    # Water-fill: bucket 114 = 50 + (365 × 1000/1100) ≈ 382;
    #             bucket 120 = 5 + (365 - 332) = 38.
    assert children[0].quantity == 382
    assert children[1].quantity == 38


def test_pov_water_fill_uniform_pov_across_children():
    """Each child's effective POV must be approximately equal under water-fill.

    Regression guard: pre-audit POV stuffed all residual into the final
    child, producing wildly unequal POV across buckets (e.g., last bucket
    at 370% POV, others at 5%). Water-fill restores uniform POV.
    """
    strat = POVStrategy(target_pov=None, cap_pov=0.05, force_completion=True)
    parent = _parent(qty=420)
    profile = _profile_simple()
    children = strat.schedule(parent, market_context={
        "volume_profile": profile, "bin_minutes": 5,
    })
    # Build per-child effective POV (qty / bucket_volume).
    bucket_vol = dict(zip(profile["bucket_min"].to_list(), profile["volume"].to_list()))
    NS_PER_MIN = 60 * 1_000_000_000
    BIN_NS = 5 * NS_PER_MIN
    pov_per_child = []
    for c in children:
        # Bucket midpoint: bucket_min × bin_ns + bin_ns/2
        # Recover bucket_min by inverting: ts = bucket_min × bin_ns + bin_ns/2
        b = (c.timestamp_ns - BIN_NS // 2) // BIN_NS
        vol = bucket_vol.get(int(b), None)
        if vol is None or vol <= 0:
            continue
        pov_per_child.append(c.quantity / vol)
    assert pov_per_child, "No child mapped to a profile bucket"
    pov_max = max(pov_per_child)
    pov_min = min(pov_per_child)
    # All within ±5% relative — uniform up to integer rounding.
    assert (pov_max - pov_min) / pov_max < 0.05, (
        f"Water-fill must equalise POV across buckets; got POVs={pov_per_child}"
    )


def test_pov_explicit_target():
    """target_pov=0.10 with cap=0.20 → 10% of each bucket volume."""
    strat = POVStrategy(target_pov=0.10, cap_pov=0.20, force_completion=False)
    # Window 09:30 to 11:00 covers buckets 114, 120, 126 fully (5-min buckets)
    parent = ParentOrder(
        ticker="AAPL", date="20200113", side="sell",
        quantity=210,
        start_ns=int(9.5 * NS_PER_HOUR),
        end_ns=int(11.0 * NS_PER_HOUR),  # extended to cover bucket 126
    )
    children = strat.schedule(parent, market_context={
        "volume_profile": _profile_simple(), "bin_minutes": 5,
    })
    qtys = [c.quantity for c in children]
    # 10% × 1000 = 100, 10% × 100 = 10, 10% × 1000 = 100 → total 210
    assert qtys == [100, 10, 100]


def test_pov_requires_profile():
    strat = POVStrategy()
    with pytest.raises(ValueError, match="volume_profile"):
        strat.schedule(_parent(), market_context={})


# ---------------------------------------------------------------------------
# Tóth strategy tests
# ---------------------------------------------------------------------------

def test_toth_validates_cap():
    from hft.strategies.toth import TothStrategy
    with pytest.raises(ValueError, match="participation_cap"):
        TothStrategy(participation_cap=0.0)
    with pytest.raises(ValueError, match="participation_cap"):
        TothStrategy(participation_cap=1.5)


def test_toth_no_cap_binding_equals_vwap_following():
    """When participation_cap is loose, Tóth = VWAP-following allocation."""
    from hft.strategies.toth import TothStrategy
    strat = TothStrategy(participation_cap=0.50)  # loose cap
    parent = ParentOrder(
        ticker="AAPL", date="20200113", side="sell",
        quantity=210,
        start_ns=int(9.5 * NS_PER_HOUR),
        end_ns=int(11.0 * NS_PER_HOUR),
    )
    children = strat.schedule(parent, market_context={
        "volume_profile": _profile_simple(), "bin_minutes": 5,
    })
    qtys = [c.quantity for c in children]
    # 1000+100+1000=2100; parent=210 → 10% of each: 100, 10, 100
    assert qtys == [100, 10, 100]


def test_toth_cap_binding_water_fills():
    """When cap binds, residual redistributed by water-filling, not stuffed at end."""
    from hft.strategies.toth import TothStrategy
    # Cap=5% means bucket 1 capped at 50, bucket 2 at 5, bucket 3 at 50.
    # Parent 200 with VWAP-prop allocation 95.2 / 9.5 / 95.2 → bucket 1 and 3 cap binds.
    strat = TothStrategy(participation_cap=0.05)
    parent = ParentOrder(
        ticker="AAPL", date="20200113", side="sell",
        quantity=200,
        start_ns=int(9.5 * NS_PER_HOUR),
        end_ns=int(11.0 * NS_PER_HOUR),
    )
    children = strat.schedule(parent, market_context={
        "volume_profile": _profile_simple(), "bin_minutes": 5,
    })
    qtys = [c.quantity for c in children]
    # Total must equal parent qty
    assert sum(qtys) == 200
    # If all caps bind hard, residual goes to bucket with biggest volume (bucket 1 or 3, both 1000)
    # The largest qty should be > cap_qty (50) since residual stuffed somewhere
    assert max(qtys) > 50


def test_toth_expected_cost():
    from hft.strategies.toth import TothStrategy
    strat = TothStrategy(Y=1.0, sigma_bps=10.0)
    cost = strat.expected_cost_bps(parent_qty=10_000, total_volume=1_000_000)
    # √(10k / 1M) = √0.01 = 0.1, × 1.0 × 10 = 1.0 bps
    assert abs(cost - 1.0) < 0.01


# ---------------------------------------------------------------------------
# CVXPY constrained AC tests
# ---------------------------------------------------------------------------

def test_cvxpy_validates_constraints():
    from hft.strategies.cvxpy_optimal import CVXPYConstrainedAC
    with pytest.raises(ValueError, match="eta"):
        CVXPYConstrainedAC(eta=0)
    with pytest.raises(ValueError, match="lambda_risk"):
        CVXPYConstrainedAC(lambda_risk=-1.0)
    with pytest.raises(ValueError, match="pov_cap"):
        CVXPYConstrainedAC(pov_cap=1.5)


def test_cvxpy_unconstrained_risk_neutral_equals_twap():
    """Without caps and λ=0, CVXPY linear-impact AC ≡ TWAP (uniform)."""
    from hft.strategies.cvxpy_optimal import CVXPYConstrainedAC
    strat = CVXPYConstrainedAC(num_slices=10, eta=1.0, lambda_risk=0.0)
    parent = ParentOrder(
        ticker="AAPL", date="20200113", side="sell",
        quantity=1000,
        start_ns=int(9.5 * NS_PER_HOUR),
        end_ns=int(10.5 * NS_PER_HOUR),
    )
    children = strat.schedule(parent, market_context={"adv_shares": 1e7})
    qtys = [c.quantity for c in children]
    # All ~equal (TWAP)
    assert max(qtys) - min(qtys) <= 1, f"Expected ~uniform, got {qtys}"
    assert sum(qtys) == 1000


def test_cvxpy_pov_cap_binds():
    """Tight POV cap forces solver to spread orders to obey constraint."""
    from hft.strategies.cvxpy_optimal import CVXPYConstrainedAC
    # 10 slices each with V=100 → cap 0.05 means q_t ≤ 5, total possible 50.
    # parent=40 → feasible.
    strat = CVXPYConstrainedAC(num_slices=10, eta=1.0, pov_cap=0.05)
    parent = ParentOrder(
        ticker="AAPL", date="20200113", side="sell",
        quantity=40,
        start_ns=int(9.5 * NS_PER_HOUR),
        end_ns=int(10.5 * NS_PER_HOUR),
    )
    profile = pl.DataFrame({
        # 10 bins between 09:30 and 10:30 → 6-min each? actually our bin is fixed 5min
        # Use 12 buckets of 5-min each from 09:30 to 10:30
        "bucket_min": [114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125],
        "volume": [100] * 12,
    })
    children = strat.schedule(parent, market_context={
        "volume_profile": profile, "bin_minutes": 5,
    })
    qtys = [c.quantity for c in children]
    assert sum(qtys) == 40
    # Each slice qty ≤ 5 (cap) — let some +/- rounding tolerance
    assert max(qtys) <= 6


def test_cvxpy_infeasible_raises():
    """Infeasible POV cap → explicit error (NO Silent Fallback)."""
    from hft.strategies.cvxpy_optimal import CVXPYConstrainedAC
    strat = CVXPYConstrainedAC(num_slices=10, eta=1.0, pov_cap=0.001)  # too tight
    parent = ParentOrder(
        ticker="AAPL", date="20200113", side="sell",
        quantity=10_000,  # too much
        start_ns=int(9.5 * NS_PER_HOUR),
        end_ns=int(10.5 * NS_PER_HOUR),
    )
    profile = pl.DataFrame({
        "bucket_min": [114, 115, 116],
        "volume": [100, 100, 100],
    })
    with pytest.raises(ValueError, match="infeasible|Loosen"):
        strat.schedule(parent, market_context={
            "volume_profile": profile, "bin_minutes": 5,
        })
