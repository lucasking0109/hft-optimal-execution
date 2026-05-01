"""ExecutionEnv — Gymnasium env for RL-based optimal execution (Phase 5E).

Two modes:
  - 'synthetic': reset() picks random pre-generated ABIDES episode (Stage 4
    cell 404 anchor; ~183 episodes available across AAPL + multi-ticker)
  - 'real_replay': reset() picks random 5-min slice from real TAQ tick data

Action: trade_rate ∈ [0, 1] — fraction of remaining inventory to execute this step.
Observation (5-dim, mid-derived only — symmetric across both modes):
  0. inventory_pct_left  ∈ [0, 1]
  1. time_pct_left       ∈ [0, 1]
  2. recent_vol_bps      = std of last 30s log-returns × 1e4 (clipped [0, 50])
  3. mid_drift_60s_bps   = (mid_now - mid_60s_ago) / mid_60s_ago × 1e4 (clipped [-50, 50])
  4. schedule_lag        = (planned_pct_executed - actual_pct_executed) ∈ [-1, 1]
                            positive = behind schedule

Reward (per step):
  reward = -step_is_bps  where step_is_bps follows metrics.py IS convention
  (positive bps = cost / bad for executor). The negation makes PPO's
  argmax(reward) align with min(IS) = good execution.
  For SELL: higher fill_price → step_is_bps negative → reward positive → RL
            maximizes → wants high fills.
  For BUY:  lower fill_price → step_is_bps negative → reward positive → RL
            maximizes → wants low fills.

  Episode-sum of step_is_bps reproduces the parent's IS_bps (qty-weighted
  integral). reward then equals -IS_bps minus terminal completion penalty.

Terminal penalty:
  When episode ends (terminated or truncated) with inventory_left > 0:
    reward -= 100 × inventory_pct_left  (force completion)

Episode:
  60 steps × 5 sec = 300 sec (5 min) horizon.
  Each step advances 5 sec in mid_prices array (i.e., index += 5).

Caveats (NO Silent Fallback — documented):
  1. No spread modelling — fill_price = mid (v1 simplification). Real fills
     should pay half-spread; this biases RL toward over-aggressive policies.
     Phase 6+ should add fixed spread cost (e.g., 0.5 bps per fill).
  2. No self-impact — child orders don't push mid (matches backtest engine).
     Caveat: RL trained on synth may over-estimate real OOS performance.
  3. Synthetic episodes have known 5-min calibration gaps (kurt over-large,
     autocorr_lag_10 over-large, trade_size under-small) — see
     `reports/multi_ticker_validation.md`. Env doesn't fix these; OOS eval
     judges impact.
  4. real_replay 5-min slice is randomly sampled — may include low-volume
     periods. Env doesn't filter; RL learns over varied conditions.
"""

from __future__ import annotations

from typing import Literal

import gymnasium as gym
import numpy as np

from hft.simulators.episode_loader import (
    EpisodeData,
    list_real_5min_slices,
    list_synthetic_episodes,
    load_real_episode,
    load_synthetic_episode,
)

Side = Literal["sell", "buy"]
Mode = Literal["synthetic", "real_replay"]


class ExecutionEnv(gym.Env):
    """5-min execution episode environment for RL training."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        mode: Mode = "synthetic",
        side: Side = "sell",
        total_qty: int = 100_000,
        ticker: str = "AAPL",
        date: str = "20200113",
        date_pool: list[str] | None = None,
        step_seconds: int = 5,
        n_steps: int = 60,
        max_action_per_step: float | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        if mode not in ("synthetic", "real_replay"):
            raise ValueError(f"mode must be 'synthetic' or 'real_replay', got {mode}")
        if side not in ("sell", "buy"):
            raise ValueError(f"side must be 'sell' or 'buy', got {side}")
        if total_qty <= 0:
            raise ValueError(f"total_qty must be > 0, got {total_qty}")
        if step_seconds * n_steps != 300:
            raise ValueError(
                f"step_seconds × n_steps must equal 300 (5 min), "
                f"got {step_seconds} × {n_steps} = {step_seconds * n_steps}"
            )
        if max_action_per_step is not None:
            if not (0 < max_action_per_step <= 1.0):
                raise ValueError(
                    f"max_action_per_step must be in (0, 1], got {max_action_per_step}"
                )

        self.mode = mode
        self.side = side
        self.side_sign = 1 if side == "sell" else -1
        self.total_qty = total_qty
        self.ticker = ticker
        self.date = date
        self.date_pool = list(date_pool) if date_pool else None
        self.step_seconds = step_seconds
        self.n_steps = n_steps
        self.max_action_per_step = max_action_per_step

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -50.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 50.0, 50.0, 1.0], dtype=np.float32),
            shape=(5,),
            dtype=np.float32,
        )
        # Action space upper bound matches max_action_per_step when set
        # (so PPO doesn't waste exploration above the cap).
        action_high = float(max_action_per_step) if max_action_per_step is not None else 1.0
        self.action_space = gym.spaces.Box(
            low=0.0, high=action_high, shape=(1,), dtype=np.float32,
        )

        # Episode pools
        if mode == "synthetic":
            self._synthetic_paths = list_synthetic_episodes()
            if not self._synthetic_paths:
                raise RuntimeError(
                    "No synthetic episodes found. Run scripts/run_phase5c_batch.py first."
                )
            self._real_starts_by_date: dict[str, list[int]] = {}
        else:  # real_replay
            # Build per-date slice pool. If date_pool provided, use those; else single date.
            dates_to_use = self.date_pool if self.date_pool else [date]
            self._real_starts_by_date = {}
            for d in dates_to_use:
                starts = list_real_5min_slices(ticker=ticker, date=d, step_minutes=5)
                if starts:
                    self._real_starts_by_date[d] = starts
            if not self._real_starts_by_date:
                raise RuntimeError(
                    f"No real 5-min slices found for {ticker} on any of {dates_to_use}."
                )
            self._synthetic_paths = []

        # RNG
        self._rng = np.random.default_rng(seed)

        # Episode state (set by reset)
        self._episode: EpisodeData | None = None
        self._step_idx: int = 0
        self._inventory_left: float = 0.0
        self._cumulative_filled: float = 0.0
        self._fill_price_sum: float = 0.0
        self._fill_qty_sum: float = 0.0

    # ---------------------------------------------------------------------
    # Gymnasium API
    # ---------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # Gymnasium bookkeeping (sets self._np_random)
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Pick a random episode from the pool
        if self.mode == "synthetic":
            idx = int(self._rng.integers(0, len(self._synthetic_paths)))
            path = self._synthetic_paths[idx]
            try:
                self._episode = load_synthetic_episode(path)
            except Exception as e:
                # Defensive: occasional bad parquet → skip to next
                # (NO Silent Fallback: but env reset must produce something)
                # Try another random episode up to 3 times
                for _ in range(3):
                    idx2 = int(self._rng.integers(0, len(self._synthetic_paths)))
                    try:
                        self._episode = load_synthetic_episode(
                            self._synthetic_paths[idx2]
                        )
                        break
                    except Exception:
                        continue
                else:
                    raise RuntimeError(
                        f"Failed to load synthetic episode after retries: {e}"
                    )
        else:
            # Multi-date: pick random date from pool, then random slice within
            available_dates = list(self._real_starts_by_date.keys())
            chosen_date = available_dates[
                int(self._rng.integers(0, len(available_dates)))
            ]
            starts = self._real_starts_by_date[chosen_date]
            idx = int(self._rng.integers(0, len(starts)))
            start_ns = starts[idx]
            self._episode = load_real_episode(
                ticker=self.ticker, date=chosen_date,
                start_ns=start_ns, slice_minutes=5,
            )

        self._step_idx = 0
        self._inventory_left = float(self.total_qty)
        self._cumulative_filled = 0.0
        self._fill_price_sum = 0.0
        self._fill_qty_sum = 0.0

        obs = self._compute_obs()
        info = {
            "source": self._episode.source,
            "arrival_mid": self._episode.arrival_mid,
            "total_qty": self.total_qty,
        }
        return obs, info

    def step(self, action: np.ndarray):
        if self._episode is None:
            raise RuntimeError("Must call reset() before step()")

        # 1. Clip action to [0, max_action_per_step] (cap=1.0 if not set)
        action_arr = np.asarray(action, dtype=np.float32).flatten()
        cap = self.max_action_per_step if self.max_action_per_step is not None else 1.0
        trade_rate = float(np.clip(action_arr[0], 0.0, cap))

        # 2. Compute quantity to fill this step
        qty_to_fill = trade_rate * self._inventory_left

        # 3. Fill price = current mid (no spread modelling — see caveat 1)
        current_mid_idx = self._step_idx * self.step_seconds
        if current_mid_idx >= len(self._episode.mid_prices):
            current_mid_idx = len(self._episode.mid_prices) - 1
        fill_price = float(self._episode.mid_prices[current_mid_idx])

        # 4. Update inventory + fill stats
        self._inventory_left -= qty_to_fill
        self._cumulative_filled += qty_to_fill
        self._fill_price_sum += fill_price * qty_to_fill
        self._fill_qty_sum += qty_to_fill

        # 5. Compute step reward (IS-bps contribution)
        arrival_mid = self._episode.arrival_mid
        if qty_to_fill > 0 and arrival_mid > 0:
            step_is_bps = (
                -self.side_sign
                * (fill_price - arrival_mid)
                / arrival_mid
                * 1e4
                * (qty_to_fill / self.total_qty)
            )
        else:
            step_is_bps = 0.0
        # Negate: step_is_bps follows metrics.py convention (positive = cost).
        # PPO maximizes reward, so flip sign so that good execution → positive reward.
        reward = -float(step_is_bps)

        # 6. Advance step index
        self._step_idx += 1

        # 7. Check termination conditions
        terminated = self._inventory_left <= 1e-6  # numerical tolerance
        truncated = self._step_idx >= self.n_steps

        # 8. Terminal penalty if not finished
        if (terminated or truncated) and self._inventory_left > 1e-6:
            inventory_pct_left = self._inventory_left / self.total_qty
            reward -= 100.0 * float(inventory_pct_left)

        # 9. Build observation
        obs = self._compute_obs()

        # 10. Info dict
        avg_fill = (
            self._fill_price_sum / self._fill_qty_sum
            if self._fill_qty_sum > 0 else arrival_mid
        )
        info = {
            "step_idx": self._step_idx,
            "inventory_left": self._inventory_left,
            "cumulative_filled": self._cumulative_filled,
            "qty_filled_this_step": qty_to_fill,
            "fill_price": fill_price,
            "arrival_mid": arrival_mid,
            "avg_fill_price": avg_fill,
            "step_is_bps": float(step_is_bps),
        }
        return obs, reward, terminated, truncated, info

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _compute_obs(self) -> np.ndarray:
        """Build 5-dim observation from current state + episode mid_prices."""
        if self._episode is None:
            raise RuntimeError("Episode not loaded")

        mids = self._episode.mid_prices
        current_mid_idx = min(self._step_idx * self.step_seconds, len(mids) - 1)

        # Feature 0: inventory_pct_left ∈ [0, 1]
        inventory_pct_left = self._inventory_left / self.total_qty

        # Feature 1: time_pct_left ∈ [0, 1]
        time_pct_left = max(0.0, 1.0 - self._step_idx / self.n_steps)

        # Feature 2: recent_vol_bps (std of last 30 sec log returns × 1e4)
        recent_vol_bps = 0.0
        if current_mid_idx >= 31:
            window = mids[current_mid_idx - 30: current_mid_idx + 1]
            window = window[window > 0]
            if len(window) >= 2:
                log_rets = np.diff(np.log(window))
                if len(log_rets) > 0:
                    recent_vol_bps = float(np.std(log_rets) * 1e4)
        recent_vol_bps = float(np.clip(recent_vol_bps, 0.0, 50.0))

        # Feature 3: mid_drift_60s_bps
        mid_drift_60s_bps = 0.0
        if current_mid_idx >= 60:
            mid_60s_ago = mids[current_mid_idx - 60]
            mid_now = mids[current_mid_idx]
            if mid_60s_ago > 0:
                mid_drift_60s_bps = (mid_now - mid_60s_ago) / mid_60s_ago * 1e4
        mid_drift_60s_bps = float(np.clip(mid_drift_60s_bps, -50.0, 50.0))

        # Feature 4: schedule_lag (TWAP plan reference)
        planned_pct_executed = self._step_idx / self.n_steps
        actual_pct_executed = self._cumulative_filled / self.total_qty
        schedule_lag = float(np.clip(
            planned_pct_executed - actual_pct_executed, -1.0, 1.0
        ))

        return np.array(
            [
                inventory_pct_left,
                time_pct_left,
                recent_vol_bps,
                mid_drift_60s_bps,
                schedule_lag,
            ],
            dtype=np.float32,
        )
