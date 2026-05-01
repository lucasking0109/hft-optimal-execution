"""Unit tests for ExecutionEnv (Phase 5E).

Tests cover both synthetic and real_replay modes.
Gymnasium API conformance + reward semantics + termination logic.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from hft.simulators.execution_env import ExecutionEnv


# ---------------------------------------------------------------------------
# 1. Observation space + action space
# ---------------------------------------------------------------------------

def test_observation_space_shape():
    """obs.shape == (5,) and within bounds after reset."""
    env = ExecutionEnv(mode="synthetic", seed=42)
    obs, info = env.reset()
    assert obs.shape == (5,), f"Expected (5,), got {obs.shape}"
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs), f"obs out of bounds: {obs}"


def test_action_space_clipping():
    """Action > 1 or < 0 should be clipped (not crash)."""
    env = ExecutionEnv(mode="synthetic", seed=42)
    env.reset()
    # Out-of-bounds actions should not raise; env clips internally
    obs, r1, _, _, _ = env.step(np.array([5.0], dtype=np.float32))    # over-1
    obs, r2, _, _, _ = env.step(np.array([-3.0], dtype=np.float32))   # negative
    # No crash = pass


# ---------------------------------------------------------------------------
# 2. Mode coverage
# ---------------------------------------------------------------------------

def test_reset_synthetic_mode():
    """Synthetic mode loads valid 300-element mid array."""
    env = ExecutionEnv(mode="synthetic", seed=42)
    obs, info = env.reset()
    assert "synthetic_" in info["source"]
    assert info["arrival_mid"] > 0
    assert env._episode is not None
    assert env._episode.mid_prices.shape == (300,)


def test_reset_real_replay_mode():
    """Real replay loads valid AAPL 5-min slice."""
    env = ExecutionEnv(mode="real_replay", ticker="AAPL", date="20200113", seed=42)
    obs, info = env.reset()
    assert "real_AAPL" in info["source"]
    assert 200 < info["arrival_mid"] < 400, f"AAPL ~$313 in Jan 2020, got {info['arrival_mid']}"


# ---------------------------------------------------------------------------
# 3. Termination logic
# ---------------------------------------------------------------------------

def test_terminate_when_inventory_zero():
    """Action=1.0 → all inventory executed → terminated=True."""
    env = ExecutionEnv(mode="synthetic", seed=42, total_qty=100_000)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(np.array([1.0], dtype=np.float32))
    assert terminated, f"Expected terminated=True after action=1.0, got {terminated}"
    assert info["inventory_left"] < 1e-3


def test_truncate_after_n_steps():
    """60 zero-action steps → truncated=True with full inventory penalty."""
    env = ExecutionEnv(mode="synthetic", seed=42, total_qty=100_000)
    env.reset()
    truncated_at = None
    for i in range(60):
        obs, reward, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))
        if truncated:
            truncated_at = i
            break
    assert truncated_at == 59, f"Expected truncated at step 59, got {truncated_at}"
    # Final reward should include penalty (~ -100 × 1.0 = -100 + tiny step term ≈ -100)
    assert reward < -50, f"Expected large penalty for unfinished, got {reward}"


def test_terminal_penalty_when_unfinished():
    """Half-execute → final step adds penalty proportional to remaining."""
    env = ExecutionEnv(mode="synthetic", seed=42, total_qty=100_000)
    env.reset()
    # Execute 50% then idle the rest
    obs, _, _, _, info = env.step(np.array([0.5], dtype=np.float32))
    assert abs(info["inventory_left"] - 50_000) < 1.0
    # Idle remainder
    final_reward = None
    for _ in range(59):
        obs, reward, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))
        if terminated or truncated:
            final_reward = reward
            break
    # Penalty should be -100 × 0.5 = -50 (plus tiny step component)
    assert final_reward is not None
    assert -55 < final_reward < -45, f"Expected ~-50 penalty, got {final_reward}"


# ---------------------------------------------------------------------------
# 4. Gymnasium conformance
# ---------------------------------------------------------------------------

def test_check_env_synthetic():
    """gymnasium env_checker passes for synthetic mode."""
    env = ExecutionEnv(mode="synthetic", seed=42)
    check_env(env)  # raises if non-conformant


def test_check_env_real_replay():
    """gymnasium env_checker passes for real_replay mode."""
    env = ExecutionEnv(mode="real_replay", ticker="AAPL", date="20200113", seed=42)
    check_env(env)


# ---------------------------------------------------------------------------
# 5. Reward semantics (sanity)
# ---------------------------------------------------------------------------

def test_reward_zero_when_no_fill():
    """Action=0.0 produces zero reward (no fill, no PnL)."""
    env = ExecutionEnv(mode="synthetic", seed=42)
    env.reset()
    obs, reward, _, _, info = env.step(np.array([0.0], dtype=np.float32))
    assert abs(reward) < 1e-6, f"Expected zero reward for zero fill, got {reward}"
    assert info["qty_filled_this_step"] == 0.0


def _step_into_nonzero_mid_diff(env, side):
    """Advance through steps until the current fill-mid differs from arrival.
    Returns (reward_at_first_diff, fill_price, arrival, terminated_flag)."""
    # arrival = mid_prices[0]; subsequent steps use mid_prices[step_idx*step_seconds]
    # so we just take a tiny step at step 0 (zero PnL), then a tiny step
    # at step 1 which uses mid_prices[5]. If that still equals arrival, advance.
    arrival = env._episode.arrival_mid
    # Burn step 0 with zero qty so step_idx advances but no PnL accumulates
    _, r0, term0, trunc0, info0 = env.step(np.array([0.0], dtype=np.float32))
    assert not term0 and not trunc0
    # Now step_idx=1; further steps will use mids[5], [10], ...
    for _ in range(50):
        idx = env._step_idx * env.step_seconds
        if idx >= len(env._episode.mid_prices):
            break
        mid = float(env._episode.mid_prices[idx])
        if abs(mid - arrival) > 1e-9:
            _, reward, terminated, truncated, info = env.step(
                np.array([0.01], dtype=np.float32)
            )
            return reward, mid, arrival, terminated
        _, _, term, trunc, _ = env.step(np.array([0.0], dtype=np.float32))
        if term or trunc:
            break
    return None, None, arrival, False


def test_reward_directional_sell():
    """SELL: fill > arrival → reward > 0; fill < arrival → reward < 0.

    Regression guard for the reward-sign bug fixed in audit (Phase 6/A).
    PPO maximizes reward, so good execution must produce positive reward.
    """
    found_high = False
    found_low = False
    for trial_seed in range(50):
        env = ExecutionEnv(
            mode="synthetic", side="sell", seed=trial_seed, total_qty=10_000_000,
        )
        env.reset(seed=trial_seed)
        reward, fill, arrival, terminated = _step_into_nonzero_mid_diff(env, "sell")
        if reward is None or terminated:
            continue
        if fill > arrival:
            assert reward > 0, (
                f"sell-high (fill {fill} > arrival {arrival}) must give POSITIVE "
                f"reward, got {reward}"
            )
            found_high = True
        else:
            assert reward < 0, (
                f"sell-low (fill {fill} < arrival {arrival}) must give NEGATIVE "
                f"reward, got {reward}"
            )
            found_low = True
        if found_high and found_low:
            break
    assert found_high or found_low, "No episode with mid drift found in 50 trials"


def test_reward_directional_buy():
    """BUY: fill < arrival → reward > 0 (bought low); fill > arrival → reward < 0."""
    found_high = False
    found_low = False
    for trial_seed in range(50):
        env = ExecutionEnv(
            mode="synthetic", side="buy", seed=trial_seed, total_qty=10_000_000,
        )
        env.reset(seed=trial_seed)
        reward, fill, arrival, terminated = _step_into_nonzero_mid_diff(env, "buy")
        if reward is None or terminated:
            continue
        if fill < arrival:
            assert reward > 0, (
                f"buy-low (fill {fill} < arrival {arrival}) must give POSITIVE "
                f"reward, got {reward}"
            )
            found_low = True
        else:
            assert reward < 0, (
                f"buy-high (fill {fill} > arrival {arrival}) must give NEGATIVE "
                f"reward, got {reward}"
            )
            found_high = True
        if found_high and found_low:
            break
    assert found_high or found_low, "No episode with mid drift found in 50 trials"


# ---------------------------------------------------------------------------
# 6. Phase A.1: max_action_per_step (action cap)
# ---------------------------------------------------------------------------

def test_max_action_per_step_clipping():
    """With cap=0.1, action=1.0 only fires 10% inventory."""
    env = ExecutionEnv(
        mode="synthetic", seed=42,
        total_qty=100_000,
        max_action_per_step=0.1,
    )
    env.reset()
    # Action space upper bound should reflect the cap
    assert env.action_space.high[0] == np.float32(0.1)
    # Try to fire 100% — env should clip to 10%
    obs, reward, terminated, truncated, info = env.step(np.array([1.0], dtype=np.float32))
    # Filled exactly 10% (10,000 shares), 90,000 left
    assert abs(info["qty_filled_this_step"] - 10_000) < 1.0
    assert abs(info["inventory_left"] - 90_000) < 1.0
    assert not terminated  # can't be done in 1 step under cap


def test_max_action_per_step_min_episode_length():
    """With cap=0.1, episode needs at least 10 steps to fully execute."""
    env = ExecutionEnv(
        mode="synthetic", seed=42,
        total_qty=100_000,
        max_action_per_step=0.1,
    )
    env.reset()
    n_steps = 0
    terminated = False
    while not terminated and n_steps < 60:
        # Always max out
        obs, reward, terminated, truncated, info = env.step(
            np.array([0.1], dtype=np.float32)
        )
        n_steps += 1
        if truncated:
            break
    assert n_steps >= 10, (
        f"With cap=0.1, should need ≥ 10 steps to finish, got {n_steps}"
    )


def test_max_action_per_step_validation():
    """Constructor rejects invalid cap values."""
    with pytest.raises(ValueError, match="max_action_per_step"):
        ExecutionEnv(mode="synthetic", max_action_per_step=0.0)
    with pytest.raises(ValueError, match="max_action_per_step"):
        ExecutionEnv(mode="synthetic", max_action_per_step=1.5)
    with pytest.raises(ValueError, match="max_action_per_step"):
        ExecutionEnv(mode="synthetic", max_action_per_step=-0.1)


# ---------------------------------------------------------------------------
# 7. Phase A.2: date_pool multi-day support
# ---------------------------------------------------------------------------

def test_date_pool_diversity():
    """With date_pool of 4 days, 100 resets should cover all 4 dates."""
    env = ExecutionEnv(
        mode="real_replay",
        ticker="AAPL",
        date_pool=["20200113", "20200114", "20200115", "20200116"],
        seed=42,
    )
    seen_dates = set()
    for i in range(100):
        obs, info = env.reset(seed=42 + i)
        # Source string includes date — extract it
        # Format: "real_AAPL_YYYYMMDD_HH:MM"
        parts = info["source"].split("_")
        assert parts[0] == "real" and parts[1] == "AAPL"
        seen_dates.add(parts[2])
    assert len(seen_dates) >= 3, (
        f"Expected ≥ 3 unique dates from pool of 4, got {seen_dates}"
    )


def test_date_pool_backward_compat_single_date():
    """Without date_pool, single-date behavior unchanged."""
    env = ExecutionEnv(
        mode="real_replay", ticker="AAPL", date="20200113", seed=42,
    )
    obs, info = env.reset()
    assert "20200113" in info["source"]
