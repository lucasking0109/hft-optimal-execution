"""Phase 6A — Train PPO RL agent on synthetic episodes.

Trains a PPO policy on `ExecutionEnv(mode='synthetic')` for 100k steps.
Synthetic episodes have known calibration gaps (Phase 5C / multi-ticker
validation): high kurtosis tails, mismatched autocorr, smaller trade sizes.

To prevent training instability from outlier episodes:
  - Reward clipped to ±50 bps per step (synthetic mode only) — synth has
    individual rewards in thousands of bps from extreme price moves;
    clipping stabilizes PPO without hiding the gap (gap evaluated on real OOS).

Output:
  - rl/checkpoints/ppo_synth_v0/
      ├─ model.zip                        (PPO weights)
      └─ vec_normalize.pkl                (obs/reward normalization)
  - rl/checkpoints/ppo_synth_v0/training_log.csv  (per-episode rewards)
  - reports/phase6a_ppo_synth.md
  - reports/figures/phase6a_training_curve.png
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


TOTAL_TIMESTEPS = 100_000
REWARD_CLIP_BPS = 50.0   # clip per-step reward to ±50 bps for synth training stability
CHECKPOINT_DIR = ROOT / "rl" / "checkpoints" / "ppo_synth_v0"


class RewardClipWrapper(gym.Wrapper):
    """Clip per-step reward to [-clip, +clip]. Documented in caveats."""
    def __init__(self, env, clip: float = REWARD_CLIP_BPS):
        super().__init__(env)
        self.clip = clip
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Save unclipped for diagnostics
        info["reward_raw"] = reward
        return obs, float(np.clip(reward, -self.clip, self.clip)), terminated, truncated, info


class TrainingLogCallback(BaseCallback):
    """Track per-episode total reward (unclipped) for plotting."""
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.current_reward = 0.0
        self.current_length = 0

    def _on_step(self) -> bool:
        # SB3 passes infos in self.locals
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", np.array([0.0]))
        dones = self.locals.get("dones", np.array([False]))
        for i, info in enumerate(infos):
            self.current_reward += float(info.get("reward_raw", rewards[i]))
            self.current_length += 1
            if dones[i]:
                self.episode_rewards.append(self.current_reward)
                self.episode_lengths.append(self.current_length)
                self.current_reward = 0.0
                self.current_length = 0
                if len(self.episode_rewards) % 100 == 0:
                    last100 = self.episode_rewards[-100:]
                    print(f"  [step {self.num_timesteps}] episodes={len(self.episode_rewards)}, "
                          f"last 100 mean={np.mean(last100):+.2f} bps, "
                          f"median={np.median(last100):+.2f}")
        return True


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    reports_dir = ROOT / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase 6A — PPO baseline training on synthetic episodes")
    print(f"Mode: synthetic ExecutionEnv (Stage 4 cell 404 episodes)")
    print(f"Total steps: {TOTAL_TIMESTEPS:,}")
    print(f"Reward clip: ±{REWARD_CLIP_BPS} bps (training stability)")
    print(f"Device: MPS (Apple Silicon GPU)")
    print("=" * 70)

    # ── Build env ──
    def make_env():
        env = ExecutionEnv(mode="synthetic", seed=42)
        env = RewardClipWrapper(env, clip=REWARD_CLIP_BPS)
        env = Monitor(env)
        return env

    env = make_env()

    # ── PPO with default hyperparams + MPS device ──
    # Note: SB3 uses torch.device('cpu' | 'cuda'); for MPS we manually set.
    device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}\n")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        device="cpu",  # SB3 PPO on small MlpPolicy is faster on CPU than MPS for small networks
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

    # ── Train ──
    print(f"🔬 Training PPO for {TOTAL_TIMESTEPS:,} steps...\n")
    callback = TrainingLogCallback()
    t0 = time.perf_counter()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, progress_bar=False)
    train_time = time.perf_counter() - t0
    print(f"\n✅ Training done in {train_time/60:.1f} min "
          f"({len(callback.episode_rewards)} episodes)")

    # ── Save model ──
    model_path = CHECKPOINT_DIR / "model.zip"
    model.save(str(model_path))
    print(f"💾 Wrote {model_path}")

    # ── Save training log ──
    log_path = CHECKPOINT_DIR / "training_log.json"
    log_path.write_text(json.dumps({
        "timestamp_et": dt.datetime.now(dt.timezone(dt.timedelta(hours=-5))).isoformat(),
        "total_timesteps": TOTAL_TIMESTEPS,
        "n_episodes": len(callback.episode_rewards),
        "train_time_s": train_time,
        "reward_clip_bps": REWARD_CLIP_BPS,
        "episode_rewards": callback.episode_rewards,
        "episode_lengths": callback.episode_lengths,
    }, default=str))
    print(f"💾 Wrote {log_path}")

    # ── Eval on 100 fresh synthetic episodes ──
    print("\n📊 Evaluating trained policy on 100 fresh synthetic episodes...")
    eval_env = ExecutionEnv(mode="synthetic", seed=999)
    rewards = []
    for ep_i in range(100):
        obs, _ = eval_env.reset(seed=999 + ep_i)
        ep_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_reward += reward
        rewards.append(ep_reward)
    rewards = np.array(rewards)
    print(f"  Eval median: {np.median(rewards):+.2f} bps")
    print(f"  Eval mean: {np.mean(rewards):+.2f} bps")
    print(f"  Eval std: {np.std(rewards):.2f}")
    print(f"  Eval IQR: [{np.percentile(rewards, 25):+.2f}, {np.percentile(rewards, 75):+.2f}]")

    eval_path = CHECKPOINT_DIR / "eval_results.json"
    eval_path.write_text(json.dumps({
        "n_eval_episodes": 100,
        "median": float(np.median(rewards)),
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "p25": float(np.percentile(rewards, 25)),
        "p75": float(np.percentile(rewards, 75)),
        "rewards": rewards.tolist(),
    }, indent=2))

    # ── Markdown report ──
    md = ["# Phase 6A — PPO baseline on synthetic episodes\n"]
    md.append(f"**Generated**: {dt.datetime.now(dt.timezone(dt.timedelta(hours=-5))).isoformat()} (ET)")
    md.append(f"**Total timesteps**: {TOTAL_TIMESTEPS:,}")
    md.append(f"**Episodes seen**: {len(callback.episode_rewards):,}")
    md.append(f"**Train time**: {train_time/60:.1f} min")
    md.append(f"**Reward clip**: ±{REWARD_CLIP_BPS} bps (training stability vs synth outliers)\n")
    md.append("> **Reward sign**: `reward = -step_is_bps` (PPO-friendly).")
    md.append("> Positive reward bps = good execution; multiply by −1 to read as IS cost.\n")

    md.append("## Training reward progression\n")
    if len(callback.episode_rewards) >= 100:
        first_100 = callback.episode_rewards[:100]
        last_100 = callback.episode_rewards[-100:]
        md.append(f"- First 100 episodes mean: {np.mean(first_100):+.2f} bps")
        md.append(f"- Last 100 episodes mean: {np.mean(last_100):+.2f} bps")
        md.append(f"- Improvement: {np.mean(last_100) - np.mean(first_100):+.2f} bps\n")

    md.append("## Eval on 100 fresh synthetic episodes (deterministic policy)\n")
    md.append(f"- Median: **{np.median(rewards):+.2f} bps**")
    md.append(f"- Mean: {np.mean(rewards):+.2f} bps")
    md.append(f"- IQR: [{np.percentile(rewards, 25):+.2f}, {np.percentile(rewards, 75):+.2f}]\n")

    md.append("## Caveats")
    md.append(f"- Synthetic episodes have known calibration gaps (kurt/autocorr/trade_size)")
    md.append(f"- Reward clip ±{REWARD_CLIP_BPS} bps applied during training (stability)")
    md.append(f"- True quality determined by Phase 6C OOS eval on real Day 5\n")

    md.append("## Next: Phase 6B")
    md.append("- Train PPO on real_replay mode (real-data primary)")
    md.append("- Compare vs this synth-trained agent on real OOS\n")

    md_path = reports_dir / "phase6a_ppo_synth.md"
    md_path.write_text("\n".join(md))
    print(f"\n💾 Wrote {md_path}")

    # ── Training curve figure ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Training curve (rolling mean)
        rew = np.array(callback.episode_rewards)
        if len(rew) > 50:
            rolling = np.convolve(rew, np.ones(50) / 50, mode='valid')
            ax1.plot(rolling, alpha=0.8)
            ax1.set_xlabel("Episode")
            ax1.set_ylabel("Rolling mean reward (50 ep window)")
            ax1.set_title("PPO Training Curve (synthetic)")
            ax1.grid(alpha=0.3)
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        # Eval distribution
        ax2.hist(rewards, bins=30, alpha=0.7, color='#4a90d9', edgecolor='black')
        ax2.axvline(x=np.median(rewards), color='red', linestyle='--', label=f"median={np.median(rewards):+.1f}")
        ax2.set_xlabel("Episode total reward (bps)")
        ax2.set_ylabel("Count")
        ax2.set_title("PPO Eval (100 synthetic episodes)")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        fig_path = figures_dir / "phase6a_training_curve.png"
        plt.savefig(fig_path, dpi=120)
        print(f"💾 Wrote {fig_path}")
    except Exception as e:
        print(f"⚠️  Figure failed: {e}")

    print("\n" + "=" * 70)
    print(f"✅ Phase 6A complete — model at {model_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
