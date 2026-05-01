"""Phase 6B — Train PPO RL agent on real-data primary (AAPL 2020-01-13).

Trains a PPO policy on `ExecutionEnv(mode='real_replay')`. Real episodes are
78 non-overlapping 5-min slices in RTH (09:30-16:00). Each reset picks a random
slice.

Compared to Phase 6A (synth-only):
  - Real data has SMALLER variance, lower kurtosis
  - PPO should NOT degenerate to all-aggressive (the synth issue)
  - Episode rewards typically ±5 bps not ±2000 bps
  - No reward clip needed

Output:
  - rl/checkpoints/ppo_real_v0/
      ├─ model.zip
      └─ training_log.json
  - reports/phase6b_ppo_real.md
  - reports/figures/phase6b_training_curve.png
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from hft.simulators.execution_env import ExecutionEnv  # noqa: E402


TOTAL_TIMESTEPS = 100_000   # match 6A for fair comparison
CHECKPOINT_DIR = ROOT / "rl" / "checkpoints" / "ppo_real_v0"


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
                if len(self.episode_rewards) % 50 == 0:
                    last50 = self.episode_rewards[-50:]
                    print(f"  [step {self.num_timesteps}] episodes={len(self.episode_rewards)}, "
                          f"last 50 mean={np.mean(last50):+.3f} bps, "
                          f"median={np.median(last50):+.3f}, "
                          f"std={np.std(last50):.2f}")
        return True


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    reports_dir = ROOT / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase 6B — PPO on real-data primary")
    print(f"Mode: real_replay AAPL 20200113 (78 5-min slices in RTH)")
    print(f"Total steps: {TOTAL_TIMESTEPS:,}")
    print(f"No reward clip (real episodes ±5 bps typical, no outliers)")
    print("=" * 70)

    def make_env():
        env = ExecutionEnv(mode="real_replay", ticker="AAPL", date="20200113", seed=42)
        env = Monitor(env)
        return env

    env = make_env()

    print(f"Using device: cpu (small MlpPolicy)\n")

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

    print(f"🔬 Training PPO for {TOTAL_TIMESTEPS:,} steps...\n")
    callback = TrainingLogCallback()
    t0 = time.perf_counter()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, progress_bar=False)
    train_time = time.perf_counter() - t0
    print(f"\n✅ Training done in {train_time/60:.1f} min "
          f"({len(callback.episode_rewards)} episodes)")

    # Save model
    model_path = CHECKPOINT_DIR / "model.zip"
    model.save(str(model_path))
    print(f"💾 Wrote {model_path}")

    log_path = CHECKPOINT_DIR / "training_log.json"
    log_path.write_text(json.dumps({
        "timestamp_et": dt.datetime.now(dt.timezone(dt.timedelta(hours=-5))).isoformat(),
        "total_timesteps": TOTAL_TIMESTEPS,
        "n_episodes": len(callback.episode_rewards),
        "train_time_s": train_time,
        "episode_rewards": callback.episode_rewards,
        "episode_lengths": callback.episode_lengths,
    }, default=str))
    print(f"💾 Wrote {log_path}")

    # Eval on 78 real slices (deterministic policy)
    print("\n📊 Evaluating trained policy on real_replay episodes (deterministic)...")
    eval_env = ExecutionEnv(mode="real_replay", ticker="AAPL", date="20200113", seed=999)
    rewards = []
    inventory_left_pcts = []
    for ep_i in range(78):  # all unique slices
        obs, _ = eval_env.reset(seed=999 + ep_i)
        ep_reward = 0.0
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_reward += reward
        rewards.append(ep_reward)
        inventory_left_pcts.append(info.get("inventory_left", 0) / eval_env.total_qty)
    rewards = np.array(rewards)
    inv_left = np.array(inventory_left_pcts)
    print(f"  Eval median: {np.median(rewards):+.3f} bps")
    print(f"  Eval mean: {np.mean(rewards):+.3f} bps")
    print(f"  Eval std: {np.std(rewards):.2f}")
    print(f"  Eval IQR: [{np.percentile(rewards, 25):+.3f}, {np.percentile(rewards, 75):+.3f}]")
    print(f"  Mean inventory left: {np.mean(inv_left)*100:.2f}%")
    print(f"  Episodes finished (inventory < 1%): {np.sum(inv_left < 0.01)}/{len(inv_left)}")

    eval_path = CHECKPOINT_DIR / "eval_results.json"
    eval_path.write_text(json.dumps({
        "n_eval_episodes": len(rewards),
        "median": float(np.median(rewards)),
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "p25": float(np.percentile(rewards, 25)),
        "p75": float(np.percentile(rewards, 75)),
        "mean_inventory_left_pct": float(np.mean(inv_left)),
        "rewards": rewards.tolist(),
    }, indent=2))

    # Compare vs 6A
    try:
        eval_6a = json.loads((ROOT / "rl" / "checkpoints" / "ppo_synth_v0" / "eval_results.json").read_text())
        synth_median = eval_6a["median"]
    except Exception:
        synth_median = None

    md = ["# Phase 6B — PPO real-data primary training\n"]
    md.append(f"**Generated**: {dt.datetime.now(dt.timezone(dt.timedelta(hours=-5))).isoformat()} (ET)")
    md.append(f"**Mode**: real_replay AAPL 20200113 (78 slices)")
    md.append(f"**Total timesteps**: {TOTAL_TIMESTEPS:,}")
    md.append(f"**Episodes seen**: {len(callback.episode_rewards):,}")
    md.append(f"**Train time**: {train_time/60:.1f} min\n")
    md.append("> **Reward sign**: `reward = -step_is_bps` (PPO-friendly).")
    md.append("> Positive reward bps = good execution; multiply by −1 to read as IS cost.\n")

    md.append("## Training reward progression\n")
    if len(callback.episode_rewards) >= 50:
        first_50 = callback.episode_rewards[:50]
        last_50 = callback.episode_rewards[-50:]
        md.append(f"- First 50 episodes mean: {np.mean(first_50):+.3f} bps")
        md.append(f"- Last 50 episodes mean: {np.mean(last_50):+.3f} bps")
        md.append(f"- Improvement: {np.mean(last_50) - np.mean(first_50):+.3f} bps\n")

    md.append("## Eval on 78 real_replay episodes (deterministic, in-sample)\n")
    md.append(f"- Median: **{np.median(rewards):+.3f} bps**")
    md.append(f"- Mean: {np.mean(rewards):+.3f} bps")
    md.append(f"- IQR: [{np.percentile(rewards, 25):+.3f}, {np.percentile(rewards, 75):+.3f}]")
    md.append(f"- Mean inventory left: {np.mean(inv_left)*100:.2f}%")
    md.append(f"- Completion rate (<1% left): {np.sum(inv_left < 0.01)}/{len(inv_left)}\n")

    md.append("## Comparison vs Phase 6A (synth-only)\n")
    if synth_median is not None:
        md.append(f"- Phase 6A (synth-only): median = {synth_median:+.2f} bps (degenerate aggressive)")
        md.append(f"- Phase 6B (real primary): median = {np.median(rewards):+.3f} bps")
        md.append(f"- 6B vs 6A diff: {np.median(rewards) - synth_median:+.3f} bps\n")

    md.append("## Caveats")
    md.append("- Train + eval both on 20200113 → in-sample, optimistic. Phase 6C OOS on Day 5 = real test.")
    md.append("- Single-day training; multi-day mix would broaden state distribution.\n")

    md.append("## Next: Phase 6C")
    md.append("- OOS evaluation on Day 5 (20200117) for both 6A and 6B models")
    md.append("- Compare vs TWAP, VWAP-following, AC-RA on real Day 5 slices")

    md_path = reports_dir / "phase6b_ppo_real.md"
    md_path.write_text("\n".join(md))
    print(f"\n💾 Wrote {md_path}")

    # Training curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        rew = np.array(callback.episode_rewards)
        if len(rew) > 50:
            window = min(50, len(rew) // 5)
            rolling = np.convolve(rew, np.ones(window) / window, mode='valid')
            ax1.plot(rolling, alpha=0.8)
            ax1.set_xlabel("Episode")
            ax1.set_ylabel(f"Rolling mean reward ({window} ep window)")
            ax1.set_title("PPO Training Curve (real_replay)")
            ax1.grid(alpha=0.3)
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        ax2.hist(rewards, bins=20, alpha=0.7, color='#4a90d9', edgecolor='black')
        ax2.axvline(x=np.median(rewards), color='red', linestyle='--', label=f"median={np.median(rewards):+.3f}")
        ax2.set_xlabel("Episode total reward (bps)")
        ax2.set_ylabel("Count")
        ax2.set_title("PPO Eval (78 real episodes, in-sample)")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        fig_path = figures_dir / "phase6b_training_curve.png"
        plt.savefig(fig_path, dpi=120)
        print(f"💾 Wrote {fig_path}")
    except Exception as e:
        print(f"⚠️  Figure failed: {e}")

    print("\n" + "=" * 70)
    print(f"✅ Phase 6B complete — model at {model_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
