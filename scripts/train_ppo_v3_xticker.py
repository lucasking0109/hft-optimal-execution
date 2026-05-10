"""Phase E — train ticker-agnostic PPO on a 20-ticker pool with spread cost.

Differences vs Phase D (train_ppo_v2_multihour.py):
  - Pool of 20 diverse tickers (covering ADV deciles), not 5 hand-picked
  - observation_mode='v3' (drops ticker_idx, adds log_adv_norm)
  - fill_at_spread=True (RL training pays half-spread cost; matches BacktestEngine)

Day 1-4 train pool; Day 5 held out for OOS.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from hft.simulators.execution_env import ExecutionEnv  # noqa: E402


TOTAL_TIMESTEPS = 500_000
SLICE_MIN = 60
STEP_SEC = 30
N_STEPS = 120
MAX_ACTION = 0.05
DATE_POOL = ["20200113", "20200114", "20200115", "20200116"]
CHECKPOINT_DIR = ROOT / "rl" / "checkpoints" / "ppo_v3_xticker"
TRAIN_POOL_PATH = ROOT / "data" / "processed" / "phase_e_train_pool.json"


class TrainingLogCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: list[float] = []
        self.current_reward = 0.0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", np.array([0.0]))
        dones = self.locals.get("dones", np.array([False]))
        for i in range(len(rewards)):
            self.current_reward += float(rewards[i])
            if dones[i]:
                self.episode_rewards.append(self.current_reward)
                self.current_reward = 0.0
                if len(self.episode_rewards) % 50 == 0:
                    last = self.episode_rewards[-50:]
                    print(
                        f"  [step {self.num_timesteps}] ep={len(self.episode_rewards)}, "
                        f"last 50 mean={np.mean(last):+.3f} bps, "
                        f"median={np.median(last):+.3f}, std={np.std(last):.2f}"
                    )
        return True


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    pool = json.loads(TRAIN_POOL_PATH.read_text())
    TICKER_POOL = pool["tickers"]

    print("=" * 70)
    print(f"Phase E — PPO v3 ticker-agnostic training (spread cost in fills)")
    print(f"Tickers ({len(TICKER_POOL)}): {TICKER_POOL}")
    print(f"Dates (train pool): {DATE_POOL}")
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Action cap: {MAX_ACTION}  |  fill_at_spread=True  |  obs=v3 (13-dim, log_adv_norm)")
    print("=" * 70)

    env = Monitor(ExecutionEnv(
        mode="real_replay",
        side="sell", total_qty=10_000,
        ticker=TICKER_POOL[0],
        ticker_pool=TICKER_POOL,
        date_pool=DATE_POOL,
        slice_minutes=SLICE_MIN, step_seconds=STEP_SEC, n_steps=N_STEPS,
        observation_mode="v3",
        fill_at_spread=True,
        adv_exclude_dates=["20200117"],  # NEVER let baselines see Day 5
        max_action_per_step=MAX_ACTION,
        seed=42,
    ))

    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=0,
        seed=42,
    )

    callback = TrainingLogCallback()
    print(f"\n🏃 Training PPO v3 for {TOTAL_TIMESTEPS:,} steps...\n")
    t0 = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, progress_bar=False)
    train_time = time.time() - t0

    model_path = CHECKPOINT_DIR / "model.zip"
    model.save(str(model_path))

    print(f"\n✅ Training done in {train_time/60:.1f} min "
          f"({len(callback.episode_rewards)} episodes)")
    print(f"💾 Wrote {model_path}")

    log_path = CHECKPOINT_DIR / "training_log.json"
    log_path.write_text(json.dumps({
        "timestamp_et": dt.datetime.now(dt.timezone(dt.timedelta(hours=-5))).isoformat(),
        "total_timesteps": TOTAL_TIMESTEPS,
        "n_episodes": len(callback.episode_rewards),
        "train_time_seconds": train_time,
        "ticker_pool": TICKER_POOL,
        "date_pool": DATE_POOL,
        "slice_minutes": SLICE_MIN,
        "step_seconds": STEP_SEC,
        "n_steps": N_STEPS,
        "max_action_per_step": MAX_ACTION,
        "observation_mode": "v3",
        "fill_at_spread": True,
        "episode_rewards": callback.episode_rewards,
    }, indent=2, default=str))
    print(f"💾 Wrote {log_path}")

    print("=" * 70)
    print(f"✅ Phase E.2 training done — model at {model_path}")
    print(f"Next: scripts/run_phase_e_xticker_eval.py for 97-ticker Day-5 OOS")
    print("=" * 70)


if __name__ == "__main__":
    main()
