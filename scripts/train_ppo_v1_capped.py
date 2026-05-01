"""Phase A.4 — Train PPO v1 with action cap + multi-day pool.

Fixes the Phase 6A/6B degenerate-aggressive issue:
  - max_action_per_step=0.1 → forces ≥10 steps to finish (no step-0 dump)
  - date_pool=Day1-4 → broader state distribution (vs 6B's Day-1 only)
  - 200k timesteps (vs 6B's 100k) for the bigger pool

Output:
  - rl/checkpoints/ppo_real_v1_capped/
      ├─ model.zip
      └─ training_log.json
  - reports/phase_a_rl_fixed.md
  - reports/figures/phase_a_training_curve.png
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


TOTAL_TIMESTEPS = 200_000
MAX_ACTION = 0.1                                      # cap per step
DATE_POOL = ["20200113", "20200114", "20200115", "20200116"]  # Day 1-4 train; Day 5 = OOS
CHECKPOINT_DIR = ROOT / "rl" / "checkpoints" / "ppo_real_v1_capped"


class TrainingLogCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.current_reward = 0.0
        self.current_length = 0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", np.array([0.0]))
        dones = self.locals.get("dones", np.array([False]))
        for i in range(len(dones)):
            self.current_reward += float(rewards[i])
            self.current_length += 1
            if dones[i]:
                self.episode_rewards.append(self.current_reward)
                self.episode_lengths.append(self.current_length)
                self.current_reward = 0.0
                self.current_length = 0
                if len(self.episode_rewards) % 100 == 0:
                    last100 = self.episode_rewards[-100:]
                    last_lens = self.episode_lengths[-100:]
                    print(f"  [step {self.num_timesteps}] ep={len(self.episode_rewards)}, "
                          f"last 100 mean={np.mean(last100):+.3f} bps, "
                          f"median={np.median(last100):+.3f}, "
                          f"std={np.std(last100):.2f}, "
                          f"avg_len={np.mean(last_lens):.1f} steps")
        return True


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    reports_dir = ROOT / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Phase A — PPO v1 (action_cap + multi-day pool)")
    print(f"Mode: real_replay AAPL {DATE_POOL}")
    print(f"max_action_per_step: {MAX_ACTION}")
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
    print("=" * 72)

    env = Monitor(ExecutionEnv(
        mode="real_replay",
        ticker="AAPL",
        date_pool=DATE_POOL,
        max_action_per_step=MAX_ACTION,
        seed=42,
    ))

    print(f"\nUsing device: cpu (small MlpPolicy)\n")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        device="cpu",
        seed=42,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    print(f"🔬 Training for {TOTAL_TIMESTEPS:,} steps...\n")
    callback = TrainingLogCallback()
    t0 = time.perf_counter()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, progress_bar=False)
    train_time = time.perf_counter() - t0
    print(f"\n✅ Training done in {train_time/60:.1f} min ({len(callback.episode_rewards)} episodes)")

    # Save model
    model_path = CHECKPOINT_DIR / "model.zip"
    model.save(str(model_path))
    print(f"💾 Wrote {model_path}")

    log_path = CHECKPOINT_DIR / "training_log.json"
    log_path.write_text(json.dumps({
        "timestamp_et": dt.datetime.now(dt.timezone(dt.timedelta(hours=-5))).isoformat(),
        "phase": "A",
        "config": {
            "max_action_per_step": MAX_ACTION,
            "date_pool": DATE_POOL,
            "total_timesteps": TOTAL_TIMESTEPS,
        },
        "n_episodes": len(callback.episode_rewards),
        "train_time_s": train_time,
        "episode_rewards": callback.episode_rewards,
        "episode_lengths": callback.episode_lengths,
    }, default=str))
    print(f"💾 Wrote {log_path}")

    # Quick in-sample eval (training distribution)
    print("\n📊 In-sample eval on Day 1-4 pool (deterministic)...")
    eval_env = ExecutionEnv(
        mode="real_replay",
        ticker="AAPL",
        date_pool=DATE_POOL,
        max_action_per_step=MAX_ACTION,
        seed=999,
    )
    rewards = []
    ep_lengths = []
    inv_left = []
    for ep_i in range(78 * 4):  # ~312 episodes
        obs, _ = eval_env.reset(seed=999 + ep_i)
        ep_reward = 0.0
        n = 0
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_reward += reward
            n += 1
        rewards.append(ep_reward)
        ep_lengths.append(n)
        inv_left.append(info.get("inventory_left", 0) / eval_env.total_qty)

    rewards_arr = np.array(rewards)
    print(f"  In-sample median: {np.median(rewards_arr):+.3f} bps")
    print(f"  In-sample mean:   {np.mean(rewards_arr):+.3f} bps")
    print(f"  In-sample std:    {np.std(rewards_arr):.2f}")
    print(f"  Mean episode len: {np.mean(ep_lengths):.1f} steps (cap forces ≥ 10)")
    print(f"  Mean inventory left: {np.mean(inv_left)*100:.2f}%")

    # Save eval
    (CHECKPOINT_DIR / "eval_in_sample.json").write_text(json.dumps({
        "n_eval_episodes": len(rewards),
        "median": float(np.median(rewards_arr)),
        "mean": float(np.mean(rewards_arr)),
        "std": float(np.std(rewards_arr)),
        "p25": float(np.percentile(rewards_arr, 25)),
        "p75": float(np.percentile(rewards_arr, 75)),
        "mean_episode_length": float(np.mean(ep_lengths)),
        "mean_inventory_left_pct": float(np.mean(inv_left)),
    }, indent=2))

    # Training curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        rew = np.array(callback.episode_rewards)
        if len(rew) > 50:
            window = min(100, len(rew) // 10)
            rolling = np.convolve(rew, np.ones(window) / window, mode='valid')
            ax1.plot(rolling, alpha=0.8, color='#4a90d9')
            ax1.set_xlabel("Episode")
            ax1.set_ylabel(f"Rolling mean reward ({window} ep window)")
            ax1.set_title("PPO v1 Training Curve (Day 1-4 pool, cap=0.1)")
            ax1.grid(alpha=0.3)
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        ax2.hist(rewards_arr, bins=30, alpha=0.7, color='#44aa44', edgecolor='black')
        ax2.axvline(x=np.median(rewards_arr), color='red', linestyle='--',
                    label=f"median={np.median(rewards_arr):+.3f}")
        ax2.set_xlabel("Episode total reward (bps)")
        ax2.set_ylabel("Count")
        ax2.set_title(f"In-sample eval ({len(rewards)} episodes)")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        fig_path = figures_dir / "phase_a_training_curve.png"
        plt.savefig(fig_path, dpi=120)
        print(f"💾 Wrote {fig_path}")
    except Exception as e:
        print(f"⚠️  Figure failed: {e}")

    print("\n" + "=" * 72)
    print(f"✅ Phase A.4 complete — model at {model_path}")
    print(f"Next: run scripts/eval_ppo_oos.py to compare on Day 5 OOS")
    print("=" * 72)


if __name__ == "__main__":
    main()
