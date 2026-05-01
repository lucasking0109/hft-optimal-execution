"""Phase 6C — Out-of-Sample evaluation on Day 5 (20200117).

Compares 6 strategies on real AAPL Day 5 (NEVER seen during training):

  Baseline agents (Phase 5F-style):
    1. Aggressive — fire all at step 0
    2. Lazy       — fire all at step 59
    3. TWAP       — uniform 1/n_steps each step
    4. Random     — uniform random ∈ [0, 1]

  RL agents (Phase 6 trained):
    5. PPO-6A     — trained on synthetic (degenerate aggressive)
    6. PPO-6B     — trained on real Day 1 (real-primary)

For each agent, run 78 episodes (all RTH 5-min slices on Day 5).
Compute median IS, mean IS, std, IQR, completion rate, win-rate vs TWAP.

This is the **gold-standard OOS evaluation** — proves whether RL transfers
from training to unseen data.

Outputs:
  - reports/phase6c_oos_comparison.md
  - reports/figures/phase6c_oos_distribution.png
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stable_baselines3 import PPO

from hft.simulators.execution_env import ExecutionEnv  # noqa: E402


OOS_DATE = "20200117"   # Day 5 — never used in 6A/6B training


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class AggressiveAgent:
    name = "aggressive"
    def __init__(self, rng): self.rng = rng
    def act(self, obs, step_idx, n_steps):
        return np.array([1.0 if step_idx == 0 else 0.0], dtype=np.float32)


class LazyAgent:
    name = "lazy"
    def __init__(self, rng): self.rng = rng
    def act(self, obs, step_idx, n_steps):
        return np.array([1.0 if step_idx == n_steps - 1 else 0.0], dtype=np.float32)


class TWAPAgent:
    name = "twap"
    def __init__(self, rng): self.rng = rng
    def act(self, obs, step_idx, n_steps):
        steps_left = max(1, n_steps - step_idx)
        return np.array([1.0 / steps_left], dtype=np.float32)


class RandomAgent:
    name = "random"
    def __init__(self, rng): self.rng = rng
    def act(self, obs, step_idx, n_steps):
        return np.array([self.rng.uniform(0, 1)], dtype=np.float32)


class PPOAgent:
    def __init__(self, name: str, model_path: Path):
        self.name = name
        self.model = PPO.load(str(model_path), device="cpu")
    def act(self, obs, step_idx, n_steps):
        action, _ = self.model.predict(obs, deterministic=True)
        return action


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episodes(agent, n_episodes: int = 78, seed_base: int = 1000) -> list[dict]:
    env = ExecutionEnv(mode="real_replay", ticker="AAPL", date=OOS_DATE, seed=seed_base)
    out = []
    for ep_i in range(n_episodes):
        obs, info = env.reset(seed=seed_base + ep_i)
        arrival_mid = info["arrival_mid"]
        total_reward = 0.0
        step_idx = 0
        terminated = truncated = False
        total_is_bps = 0.0
        while not (terminated or truncated):
            action = agent.act(obs, step_idx, env.n_steps)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            # Sum step_is_bps from info dict — metrics.py convention
            # (positive = cost), independent of reward sign convention.
            total_is_bps += info.get("step_is_bps", 0.0)
            step_idx += 1
        out.append({
            "episode": ep_i,
            "total_reward": total_reward,
            "total_is_bps": total_is_bps,
            "n_steps": step_idx,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "inventory_left_pct": info["inventory_left"] / env.total_qty,
            "arrival_mid": arrival_mid,
        })
    return out


def stats(results: list[dict]) -> dict:
    # Report on IS_bps (metrics.py convention: positive = cost; lower = better).
    rewards = np.array([r["total_is_bps"] for r in results])
    return {
        "n": len(results),
        "median": float(np.median(rewards)),
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "p25": float(np.percentile(rewards, 25)),
        "p75": float(np.percentile(rewards, 75)),
        "min": float(np.min(rewards)),
        "max": float(np.max(rewards)),
        "completion_rate": float(np.mean([r["inventory_left_pct"] < 0.01 for r in results])),
    }


def win_rate(results_a: list[dict], results_b: list[dict]) -> float:
    """Per-episode win rate (A beats B = A has lower IS than B)."""
    wins = 0
    for ra, rb in zip(results_a, results_b):
        if ra["total_is_bps"] < rb["total_is_bps"]:
            wins += 1
    return wins / len(results_a)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    reports_dir = ROOT / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Phase 6C — OOS Evaluation on Day 5 ({OOS_DATE})")
    print(f"6 strategies × 78 episodes (all RTH 5-min slices)")
    print("=" * 70)

    rng = np.random.default_rng(42)
    all_results: dict[str, list[dict]] = {}

    # 4 baseline agents
    for AgentCls in [AggressiveAgent, LazyAgent, TWAPAgent, RandomAgent]:
        agent = AgentCls(rng)
        print(f"\n  Running {agent.name}...")
        all_results[agent.name] = run_episodes(agent, n_episodes=78)
        s = stats(all_results[agent.name])
        print(f"    median={s['median']:+.3f}  mean={s['mean']:+.3f}  "
              f"std={s['std']:.2f}  IQR=[{s['p25']:+.3f}, {s['p75']:+.3f}]  "
              f"complete={s['completion_rate']*100:.0f}%")

    # 3 PPO agents (added v1_capped from Phase A)
    for ppo_name, ckpt in [
        ("ppo_synth", ROOT / "rl" / "checkpoints" / "ppo_synth_v0" / "model.zip"),
        ("ppo_real", ROOT / "rl" / "checkpoints" / "ppo_real_v0" / "model.zip"),
        ("ppo_v1_capped", ROOT / "rl" / "checkpoints" / "ppo_real_v1_capped" / "model.zip"),
    ]:
        if not ckpt.exists():
            print(f"  ⚠️  Missing {ckpt} — skipping {ppo_name}")
            continue
        print(f"\n  Running {ppo_name} (load {ckpt.name})...")
        agent = PPOAgent(ppo_name, ckpt)
        all_results[ppo_name] = run_episodes(agent, n_episodes=78)
        s = stats(all_results[ppo_name])
        print(f"    median={s['median']:+.3f}  mean={s['mean']:+.3f}  "
              f"std={s['std']:.2f}  IQR=[{s['p25']:+.3f}, {s['p75']:+.3f}]  "
              f"complete={s['completion_rate']*100:.0f}%")

    # Compute win rates vs TWAP
    print("\n" + "=" * 70)
    print(f"Win rates vs TWAP (per-episode IS_bps lower than TWAP):")
    twap_results = all_results["twap"]
    win_rates = {}
    for name, results in all_results.items():
        if name == "twap":
            continue
        wr = win_rate(results, twap_results)
        win_rates[name] = wr
        print(f"  {name:<15} win rate = {wr*100:.1f}%")

    # Save full results
    summary = {
        "timestamp_et": dt.datetime.now(dt.timezone(dt.timedelta(hours=-5))).isoformat(),
        "oos_date": OOS_DATE,
        "n_episodes_per_agent": 78,
        "stats": {name: stats(r) for name, r in all_results.items()},
        "win_rates_vs_twap": win_rates,
        "per_episode_is_bps": {
            name: [r["total_is_bps"] for r in results]
            for name, results in all_results.items()
        },
    }
    json_path = reports_dir / "phase6c_oos_comparison.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n💾 Wrote {json_path}")

    # Markdown report
    md = ["# Phase 6C — Out-of-Sample Evaluation (Day 5 = 2020-01-17)\n"]
    md.append(f"**Generated**: {summary['timestamp_et']} (ET)")
    md.append(f"**OOS Date**: {OOS_DATE} (NEVER seen during 6A/6B training)")
    md.append(f"**Episodes per agent**: 78 (all RTH 5-min slices)\n")

    md.append("## All-strategy comparison (median + IQR + win rate vs TWAP)\n")
    md.append("| Agent | Median IS (bps) | Mean | Std | IQR | Completion | vs TWAP win |")
    md.append("|---|---|---|---|---|---|---|")
    for name in ["aggressive", "lazy", "twap", "random", "ppo_synth", "ppo_real", "ppo_v1_capped"]:
        if name not in all_results:
            continue
        s = summary["stats"][name]
        wr = win_rates.get(name, "—")
        wr_str = f"{wr*100:.0f}%" if isinstance(wr, float) else "—"
        md.append(f"| {name} | **{s['median']:+.3f}** | {s['mean']:+.3f} | "
                  f"{s['std']:.2f} | [{s['p25']:+.3f}, {s['p75']:+.3f}] | "
                  f"{s['completion_rate']*100:.0f}% | {wr_str} |")

    md.append("\n## Key findings\n")
    md.append("Stats report **IS in bps; lower is better** (positive = cost vs arrival).\n")
    if "ppo_real" in all_results and "twap" in all_results:
        ppo_real_med = summary["stats"]["ppo_real"]["median"]
        twap_med = summary["stats"]["twap"]["median"]
        gap = ppo_real_med - twap_med  # negative = RL has lower IS = RL wins
        ppo_real_wr = win_rates.get("ppo_real", 0) * 100
        verdict = "✅ RL beats TWAP" if gap < -0.5 else \
                  "🟡 RL ≈ TWAP" if abs(gap) <= 0.5 else \
                  "🔴 TWAP beats RL"
        md.append(f"- **PPO-6B (real-trained) vs TWAP**: median diff = **{gap:+.3f} bps** "
                  f"(negative = RL lower IS), win rate {ppo_real_wr:.0f}% — {verdict}")
    if "ppo_synth" in all_results and "twap" in all_results:
        ppo_synth_med = summary["stats"]["ppo_synth"]["median"]
        gap = ppo_synth_med - summary["stats"]["twap"]["median"]
        md.append(f"- **PPO-6A (synth-trained) vs TWAP**: median diff = "
                  f"{gap:+.3f} bps")
    if "ppo_real" in all_results and "ppo_synth" in all_results:
        gap = (summary["stats"]["ppo_real"]["median"]
               - summary["stats"]["ppo_synth"]["median"])
        md.append(f"- **PPO-6B vs PPO-6A**: median diff = **{gap:+.3f} bps** "
                  f"(negative = real-trained better than synth-trained)")

    md.append("\n## Caveats\n")
    md.append("- Single OOS day (20200117); larger sample needed for statistical claims.")
    md.append("- 5-min episodes; conclusions don't extrapolate to multi-hour parent orders.")
    md.append("- AC-RA / VWAP-following baselines from Phase 3 evaluated separately on different infrastructure.")
    md.append("- No spread modelling — fills at mid (caveat from env design).\n")

    md.append("## Reference: Phase 3 backtest results (different infra, multi-day)\n")
    md.append("- AC-RA beats TWAP by **5.3 bps median IS** on 5-day OOS (Phase 3 backtest engine)")
    md.append("- This eval uses ExecutionEnv on Day 5 only — direct comparison limited.")

    md_path = reports_dir / "phase6c_oos_comparison.md"
    md_path.write_text("\n".join(md))
    print(f"💾 Wrote {md_path}")

    # Distribution plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 6))
        agents = ["aggressive", "lazy", "twap", "random", "ppo_synth", "ppo_real", "ppo_v1_capped"]
        agents = [a for a in agents if a in all_results]
        data = [[r["total_is_bps"] for r in all_results[a]] for a in agents]
        parts = ax.violinplot(data, showmedians=True, widths=0.7)
        colors = ['#888888', '#ddaa00', '#4488dd', '#aa44dd', '#dd6644', '#44aa44', '#aa66cc']
        for pc, c in zip(parts['bodies'], colors):
            pc.set_facecolor(c)
            pc.set_alpha(0.65)
        ax.set_xticks(range(1, len(agents) + 1))
        ax.set_xticklabels(agents, fontsize=10)
        ax.set_ylabel("OOS Episode IS (bps; lower = better)")
        ax.set_title(f"Phase 6C — OOS Evaluation Day 5 ({OOS_DATE}), 78 episodes")
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig_path = figures_dir / "phase6c_oos_distribution.png"
        plt.savefig(fig_path, dpi=120)
        print(f"💾 Wrote {fig_path}")
    except Exception as e:
        print(f"⚠️  Figure failed: {e}")

    print("\n" + "=" * 70)
    print("Phase 6C complete — see reports/phase6c_oos_comparison.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
