"""Phase G.4 — train PPO v4 with overlapping windows covering full RTH.

Difference vs v3 (Phase E):
  - Training pool now has 12 overlapping 60-min windows per day (every 30 min
    starting from 09:30 through 15:00), instead of v3's 6 non-overlapping
    windows starting from 09:30 through 14:30.
  - Critically, this includes 15:00-16:00 (the close hour) which v3 NEVER
    saw during training, causing the G.1 generalization failure
    (RL win-rate vs VWAP-follow in close = 25%).

Everything else identical to v3:
  - 20-ticker pool, Day 1-4 train, observation_mode='v3' (13-dim ticker-agnostic),
    fill_at_spread=True, max_action_per_step=0.05, 500k timesteps.
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
SLICE_MIN = 60                # episode length
WINDOW_STEP_MIN = 30          # step between window starts → OVERLAPPING
STEP_SEC = 30
N_STEPS = 120
MAX_ACTION = 0.05
DATE_POOL = ["20200113", "20200114", "20200115", "20200116"]
CHECKPOINT_DIR = ROOT / "rl" / "checkpoints" / "ppo_v4_overlapping"
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
    print(f"Phase G.4 — PPO v4 overlapping-windows training")
    print(f"Tickers ({len(TICKER_POOL)}): {TICKER_POOL}")
    print(f"Dates: {DATE_POOL}")
    print(f"Windows: 60-min episodes, step {WINDOW_STEP_MIN} min "
          f"→ 12 overlapping windows / day covering full RTH (09:30-16:00)")
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
    print("=" * 70)

    env = Monitor(ExecutionEnv(
        mode="real_replay",
        side="sell", total_qty=10_000,
        ticker=TICKER_POOL[0],
        ticker_pool=TICKER_POOL,
        date_pool=DATE_POOL,
        slice_minutes=SLICE_MIN,
        window_step_minutes=WINDOW_STEP_MIN,   # KEY: overlapping windows
        step_seconds=STEP_SEC, n_steps=N_STEPS,
        observation_mode="v3",
        fill_at_spread=True,
        adv_exclude_dates=["20200117"],   # never see Day 5
        max_action_per_step=MAX_ACTION,
        seed=42,
    ))

    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=2048, batch_size=64, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        verbose=0, seed=42,
    )

    callback = TrainingLogCallback()
    print(f"\n🏃 Training PPO v4 for {TOTAL_TIMESTEPS:,} steps...\n")
    t0 = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback,
                progress_bar=False)
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
        "window_step_minutes": WINDOW_STEP_MIN,
        "step_seconds": STEP_SEC, "n_steps": N_STEPS,
        "max_action_per_step": MAX_ACTION,
        "observation_mode": "v3", "fill_at_spread": True,
        "episode_rewards": callback.episode_rewards,
    }, indent=2, default=str))
    print(f"💾 Wrote {log_path}")

    print("=" * 70)
    print(f"✅ Phase G.4 training done — model at {model_path}")
    print(f"Next: re-run scripts/run_phase_g_multi_window.py with v4")
    print("=" * 70)


if __name__ == "__main__":
    main()
