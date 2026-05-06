"""Phase D — train PPO on 60-min episodes with the v2 microstructure obs.

Multi-ticker × multi-day pool (Day 1-4 train; Day 5 reserved for OOS).
500k timesteps, action_cap=0.05, observation_mode="v2".

Saves to rl/checkpoints/ppo_v2_multihour/.
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
TICKER_POOL = ["AAPL", "AMZN", "AMD", "TSLA", "NVDA"]
DATE_POOL = ["20200113", "20200114", "20200115", "20200116"]  # Day 1-4 train
CHECKPOINT_DIR = ROOT / "rl" / "checkpoints" / "ppo_v2_multihour"


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
    print("=" * 70)
    print(f"Phase D — PPO 60-min multi-hour training (v2 microstructure obs)")
    print(f"Tickers: {TICKER_POOL}")
    print(f"Dates (train pool): {DATE_POOL}")
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Action cap: {MAX_ACTION}")
    print("=" * 70)

    env = Monitor(ExecutionEnv(
        mode="real_replay",
        side="sell", total_qty=10_000,
        ticker=TICKER_POOL[0],   # required by env API; ticker_pool overrides
        ticker_pool=TICKER_POOL,
        date_pool=DATE_POOL,
        slice_minutes=SLICE_MIN, step_seconds=STEP_SEC, n_steps=N_STEPS,
        observation_mode="v2",
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
    print(f"\n🏃 Training PPO for {TOTAL_TIMESTEPS:,} steps...\n")
    t0 = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, progress_bar=False)
    train_time = time.time() - t0

    model_path = CHECKPOINT_DIR / "model.zip"
    model.save(str(model_path))

    print(f"\n✅ Training done in {train_time/60:.1f} min "
          f"({len(callback.episode_rewards)} episodes)")
    print(f"💾 Wrote {model_path}")

    # Persist training log
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
        "observation_mode": "v2",
        "episode_rewards": callback.episode_rewards,
    }, indent=2, default=str))
    print(f"💾 Wrote {log_path}")

    # In-sample eval (Day 1-4, deterministic policy)
    print("\n📊 In-sample eval on Day 1-4 pool (deterministic)...")
    eval_env = ExecutionEnv(
        mode="real_replay",
        side="sell", total_qty=10_000,
        ticker=TICKER_POOL[0],
        ticker_pool=TICKER_POOL,
        date_pool=DATE_POOL,
        slice_minutes=SLICE_MIN, step_seconds=STEP_SEC, n_steps=N_STEPS,
        observation_mode="v2",
        max_action_per_step=MAX_ACTION,
        seed=999,
    )
    rewards = []
    inv_left = []
    for ep in range(60):
        obs, _ = eval_env.reset(seed=999 + ep)
        ep_r = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = eval_env.step(action)
            ep_r += r
            if term or trunc:
                break
        rewards.append(ep_r)
        inv_left.append(info["inventory_left"] / eval_env.total_qty)

    rewards_arr = np.array(rewards)
    inv_arr = np.array(inv_left)
    print(f"  In-sample median reward: {np.median(rewards_arr):+.3f}")
    print(f"  In-sample mean reward:   {np.mean(rewards_arr):+.3f}")
    print(f"  In-sample std:           {np.std(rewards_arr):.2f}")
    print(f"  Mean inventory left:     {np.mean(inv_arr)*100:.2f}%")

    eval_path = CHECKPOINT_DIR / "eval_in_sample.json"
    eval_path.write_text(json.dumps({
        "n_eval_episodes": len(rewards),
        "median_reward": float(np.median(rewards_arr)),
        "mean_reward": float(np.mean(rewards_arr)),
        "std_reward": float(np.std(rewards_arr)),
        "mean_inventory_left_pct": float(np.mean(inv_arr)),
        "rewards": rewards_arr.tolist(),
    }, indent=2))
    print(f"💾 Wrote {eval_path}")

    print("=" * 70)
    print(f"✅ Phase D training done — model at {model_path}")
    print(f"Next: scripts/run_phase_d_eval.py for Day 5 OOS comparison")
    print("=" * 70)


if __name__ == "__main__":
    main()
