"""Phase E — mini 4-fold walk-forward CV.

Folds:
  k=1: train on Day 1            → eval on Day 2
  k=2: train on Day 1-2          → eval on Day 3
  k=3: train on Day 1-2-3        → eval on Day 4
  k=4: train on Day 1-2-3-4      → eval on Day 5

Each fold:
  1. Trains a fresh PPO v3 policy (500k timesteps) on the train days.
  2. Evaluates on a sample of tickers from the full pool on the next day.
  3. Reports median IS + IS std vs vwap_following baseline.

5 days is little; fold k=1 has only 1 day of training data so the policy
will be weak. The point is **variance estimation across folds**, not a
single point estimate.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from hft.analysis.impact import (                                 # noqa: E402
    ETA_PRIOR_BPS_PER_PCT_ADV,
    GAMMA_PRIOR_BPS_PER_PCT_ADV,
)
from hft.backtest.engine import BacktestEngine                   # noqa: E402
from hft.data import load_eq_taq                                 # noqa: E402
from hft.simulators.adv_cache import get_adv                     # noqa: E402
from hft.simulators.execution_env import ExecutionEnv            # noqa: E402
from hft.strategies.almgren_chriss import AlmgrenChrissStrategy  # noqa: E402
from hft.strategies.base import ParentOrder                      # noqa: E402
from hft.strategies.rl_agent import RLAgentStrategy              # noqa: E402
from hft.strategies.twap import TWAPStrategy                     # noqa: E402
from hft.strategies.vwap_following import VWAPFollowingStrategy  # noqa: E402

DATES = ["20200113", "20200114", "20200115", "20200116", "20200117"]
TOTAL_TIMESTEPS = 500_000
SLICE_MIN = 60
STEP_SEC = 30
N_STEPS = 120
MAX_ACTION = 0.05
PARENT_QTY = 10_000
START_TIME_NS = 10 * 3600 * int(1e9)
END_TIME_NS = 11 * 3600 * int(1e9)
WALKFORWARD_DIR = ROOT / "rl" / "checkpoints" / "ppo_v3_walkforward"
TRAIN_POOL_PATH = ROOT / "data" / "processed" / "phase_e_train_pool.json"
TAQ_ROOT = ROOT / "data" / "processed" / "eq_taq"

# For eval speed: use a curated 20-ticker subset (not all 97) per fold.
# Same tickers across folds so cross-fold variance is comparable.
EVAL_POOL_SIZE = 20


def all_5_days(t: str) -> bool:
    return all((TAQ_ROOT / d / f"{t}.parquet").exists() for d in DATES)


def list_all_tickers() -> list[str]:
    candidates = sorted(
        f.replace(".parquet", "")
        for f in os.listdir(TAQ_ROOT / DATES[0]) if f.endswith(".parquet")
    )
    return [t for t in candidates if all_5_days(t)]


class SilentCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards: list[float] = []
        self._cur = 0.0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", np.array([0.0]))
        dones = self.locals.get("dones", np.array([False]))
        for i in range(len(rewards)):
            self._cur += float(rewards[i])
            if dones[i]:
                self.episode_rewards.append(self._cur)
                self._cur = 0.0
        return True


def train_fold(k: int, train_dates: list[str], train_pool: list[str]) -> Path:
    """Train a fresh PPO v3 on the given train_dates. Returns model path."""
    fold_dir = WALKFORWARD_DIR / f"fold_{k}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    model_path = fold_dir / "model.zip"
    if model_path.exists():
        print(f"  fold {k}: model already exists at {model_path}, skipping retrain")
        return model_path

    env = Monitor(ExecutionEnv(
        mode="real_replay", side="sell", total_qty=PARENT_QTY,
        ticker=train_pool[0], ticker_pool=train_pool,
        date_pool=train_dates,
        slice_minutes=SLICE_MIN, step_seconds=STEP_SEC, n_steps=N_STEPS,
        observation_mode="v3",
        fill_at_spread=True,
        adv_exclude_dates=["20200117"],  # never let any baseline see Day 5
        max_action_per_step=MAX_ACTION,
        seed=42 + k,
    ))
    model = PPO("MlpPolicy", env, learning_rate=3e-4,
                n_steps=2048, batch_size=64, n_epochs=10,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                verbose=0, seed=42 + k)
    print(f"  fold {k}: training PPO on {train_dates} ({TOTAL_TIMESTEPS:,} steps)...")
    t0 = time.time()
    callback = SilentCallback()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, progress_bar=False)
    model.save(str(model_path))
    print(f"  fold {k}: trained in {(time.time()-t0)/60:.1f} min "
          f"({len(callback.episode_rewards)} eps, "
          f"last-50 median = {np.median(callback.episode_rewards[-50:]):+.3f})")
    return model_path


def eval_fold(k: int, test_date: str, model_path: Path, eval_pool: list[str]) -> dict:
    """Evaluate RL vs TWAP / AC-RA / VWAP-follow on the test date.
    Returns a dict of per-strategy median IS + RL vs VWAP delta.
    """
    rl_strat = RLAgentStrategy(
        model_path=str(model_path),
        slice_minutes=SLICE_MIN, step_seconds=STEP_SEC, n_steps=N_STEPS,
        observation_mode="v3", max_action_per_step=MAX_ACTION,
    )
    rl_strat.name = "rl_v3"
    rows = []
    for ticker in eval_pool:
        try:
            df = load_eq_taq(ticker, test_date)
            engine = BacktestEngine(ticker, test_date, market_df=df)
            ctx = engine.market_context(bin_minutes=5)
            ctx["adv_shares"] = get_adv(ticker, exclude_dates=["20200117"])
        except Exception:
            continue
        parent = ParentOrder(
            ticker=ticker, date=test_date, side="sell",
            quantity=PARENT_QTY, start_ns=START_TIME_NS, end_ns=END_TIME_NS,
        )
        eta = ETA_PRIOR_BPS_PER_PCT_ADV.get(
            ticker, ETA_PRIOR_BPS_PER_PCT_ADV["_default"])
        gamma = GAMMA_PRIOR_BPS_PER_PCT_ADV.get(
            ticker, GAMMA_PRIOR_BPS_PER_PCT_ADV["_default"])
        adv = ctx["adv_shares"]
        strats = [
            ("twap",           TWAPStrategy(num_slices=60)),
            ("ac_ra",          AlmgrenChrissStrategy(
                num_slices=60, eta_bps_per_pct_adv=eta,
                gamma_bps_per_pct_adv=gamma, sigma_bps_per_sqrt_sec=1.0,
                lambda_risk=1e-3, adv_shares=adv,
            )),
            ("vwap_following", VWAPFollowingStrategy(num_slices=60)),
            ("rl_v3",          rl_strat),
        ]
        for label, strat in strats:
            try:
                res = engine.run(parent, strat, market_context=ctx)
                rows.append({"ticker": ticker, "strategy": label,
                             "is_bps": res.metrics.is_bps})
            except Exception as e:
                print(f"    ❌ {ticker} {label}: {type(e).__name__}: {str(e)[:60]}")
    return {"rows": rows, "n_done": len(rows)}


def main():
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    WALKFORWARD_DIR.mkdir(parents=True, exist_ok=True)

    train_pool = json.loads(TRAIN_POOL_PATH.read_text())["tickers"]
    all_tickers = list_all_tickers()
    # Eval pool: deterministic sample of EVAL_POOL_SIZE tickers from the full set
    rng = np.random.default_rng(123)
    eval_pool = sorted(rng.choice(all_tickers, size=min(EVAL_POOL_SIZE, len(all_tickers)),
                                  replace=False).tolist())
    print("=" * 70)
    print(f"Phase E.4 — Walk-forward CV (4 folds)")
    print(f"Train pool ({len(train_pool)}): {train_pool}")
    print(f"Eval pool ({len(eval_pool)}): {eval_pool}")
    print("=" * 70)

    all_rows = []
    fold_summary = []
    for k in range(1, 5):
        train_dates = DATES[:k]
        test_date = DATES[k]
        print(f"\n=== Fold {k}: train {train_dates} → test {test_date} ===")
        model_path = train_fold(k, train_dates, train_pool)
        result = eval_fold(k, test_date, model_path, eval_pool)
        for r in result["rows"]:
            r["fold"] = k
            r["test_date"] = test_date
        all_rows.extend(result["rows"])

        # Per-fold summary
        df = pl.DataFrame(result["rows"])
        agg = df.group_by("strategy").agg(pl.col("is_bps").median().alias("median_is"))
        med = {r["strategy"]: r["median_is"] for r in agg.iter_rows(named=True)}
        rl_vs_vwap = med.get("rl_v3", float("nan")) - med.get("vwap_following", float("nan"))
        rl_vs_twap = med.get("rl_v3", float("nan")) - med.get("twap", float("nan"))
        fold_summary.append({
            "fold": k, "train_dates": train_dates, "test_date": test_date,
            "n_tickers_done": len({r["ticker"] for r in result["rows"]}),
            "median_is_rl_v3": med.get("rl_v3"),
            "median_is_vwap_following": med.get("vwap_following"),
            "median_is_ac_ra": med.get("ac_ra"),
            "median_is_twap": med.get("twap"),
            "rl_minus_vwap_bps": rl_vs_vwap,
            "rl_minus_twap_bps": rl_vs_twap,
        })
        print(f"  Fold {k} medians: RL={med.get('rl_v3'):+.2f}  "
              f"vwap_follow={med.get('vwap_following'):+.2f}  "
              f"ac_ra={med.get('ac_ra'):+.2f}  twap={med.get('twap'):+.2f}")
        print(f"  RL − VWAP = {rl_vs_vwap:+.2f} bps")

    # Persist full results
    df_all = pl.DataFrame(all_rows)
    csv_path = reports_dir / "phase_e_walkforward.csv"
    df_all.write_csv(csv_path)
    print(f"\n💾 Wrote {csv_path}")

    # Markdown
    md = ["# Phase E.4 — Mini Walk-Forward CV (4 folds)\n"]
    md.append(f"**Generated**: {dt.datetime.now(dt.timezone(dt.timedelta(hours=-5))).isoformat()} (ET)")
    md.append(f"**Eval pool**: {len(eval_pool)} tickers (`{', '.join(eval_pool)}`)")
    md.append(f"**Train pool**: {len(train_pool)} tickers (same across folds)")
    md.append("**Parent**: 10,000 shares sell, 10:00–11:00 ET, NBBO routing, spread-cost fills\n")

    md.append("## Per-fold median IS\n")
    md.append("| Fold | Train days | Test day | n tickers | RL | VWAP-follow | AC-RA | TWAP | RL − VWAP |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for f in fold_summary:
        md.append(
            f"| {f['fold']} | {', '.join(f['train_dates'])} | {f['test_date']} | "
            f"{f['n_tickers_done']} | "
            f"{f['median_is_rl_v3']:+.2f} | {f['median_is_vwap_following']:+.2f} | "
            f"{f['median_is_ac_ra']:+.2f} | {f['median_is_twap']:+.2f} | "
            f"**{f['rl_minus_vwap_bps']:+.2f}** |"
        )

    # Cross-fold aggregate
    deltas = [f["rl_minus_vwap_bps"] for f in fold_summary]
    md.append(f"\n**Cross-fold RL − VWAP deltas**: {[f'{d:+.2f}' for d in deltas]}")
    md.append(f"- mean: **{np.mean(deltas):+.3f} bps**")
    md.append(f"- median: {np.median(deltas):+.3f} bps")
    md.append(f"- std across folds: {np.std(deltas):.3f} bps")

    md.append("\n## Limitations\n")
    md.append("- Fold 1 trains on only 1 day. Policy is severely data-limited; treat as lower bound.")
    md.append("- Folds 2-4 progressively gain training data. Fold 4 is closest to Phase E.3 setup.")
    md.append("- Eval pool size is small to keep runtime manageable; not representative of full 97-ticker universe.")

    md_path = reports_dir / "phase_e_walkforward.md"
    md_path.write_text("\n".join(md))
    print(f"💾 Wrote {md_path}")

    # JSON summary for downstream consumption
    summary_path = reports_dir / "phase_e_walkforward_summary.json"
    summary_path.write_text(json.dumps(fold_summary, indent=2, default=str))
    print(f"💾 Wrote {summary_path}")

    print("\n" + "=" * 70)
    print("Phase E.4 walk-forward done — see reports/phase_e_walkforward.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
