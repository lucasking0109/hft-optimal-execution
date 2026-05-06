"""Phase D — OOS evaluation: 60-min RL vs 5 Phase-B baselines on Day 5.

Same parent spec as Phase B.7 (10k shares sell, 10:00–11:00 ET, NBBO routing)
so the verdict is directly comparable. RL-v2 is loaded from the checkpoint
trained by scripts/train_ppo_v2_multihour.py.

Strategies evaluated:
  1. TWAP                  — uniform schedule
  2. AC-RA                 — Phase B's multi-metric winner
  3. VWAP-following        — volume-aware
  4. POV (5% cap)          — volume-aware with cap
  5. Tóth                  — square-root impact, water-filled
  6. RL-v2                 — PPO with 13-dim microstructure obs

Across 5 tickers × 5 days × 6 strategies. Outputs:
  - reports/phase_d_rl_v2.csv
  - reports/phase_d_rl_v2.md
  - reports/figures/phase_d_rl_v2_distribution.png
"""

from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hft.analysis.impact import (                                 # noqa: E402
    ETA_PRIOR_BPS_PER_PCT_ADV,
    GAMMA_PRIOR_BPS_PER_PCT_ADV,
    average_daily_volume_from_history,
)
from hft.backtest.engine import BacktestEngine                   # noqa: E402
from hft.data import load_eq_taq                                 # noqa: E402
from hft.strategies.almgren_chriss import AlmgrenChrissStrategy  # noqa: E402
from hft.strategies.base import ParentOrder                      # noqa: E402
from hft.strategies.pov import POVStrategy                       # noqa: E402
from hft.strategies.rl_agent import RLAgentStrategy              # noqa: E402
from hft.strategies.toth import TothStrategy                     # noqa: E402
from hft.strategies.twap import TWAPStrategy                     # noqa: E402
from hft.strategies.vwap_following import VWAPFollowingStrategy  # noqa: E402

TICKERS = ["AAPL", "AMZN", "AMD", "TSLA", "NVDA"]
DATES = ["20200113", "20200114", "20200115", "20200116", "20200117"]
NUM_SLICES = 60
LAMBDA_RISK = 1e-3
SIGMA_DEFAULT = 1.0
PARENT_QTY = 10_000
START_TIME_NS = 10 * 3600 * int(1e9)
END_TIME_NS = 11 * 3600 * int(1e9)
RL_MODEL_PATH = ROOT / "rl" / "checkpoints" / "ppo_v2_multihour" / "model.zip"

ADV_CACHE: dict[str, float] = {}


def get_adv(ticker: str) -> float:
    if ticker not in ADV_CACHE:
        try:
            ADV_CACHE[ticker] = average_daily_volume_from_history(ticker, lookback_days=60)
        except Exception:
            ADV_CACHE[ticker] = 1e7
    return ADV_CACHE[ticker]


def make_strategies(ticker: str) -> list[tuple[str, object]]:
    eta = ETA_PRIOR_BPS_PER_PCT_ADV.get(ticker, ETA_PRIOR_BPS_PER_PCT_ADV["_default"])
    gamma = GAMMA_PRIOR_BPS_PER_PCT_ADV.get(ticker, GAMMA_PRIOR_BPS_PER_PCT_ADV["_default"])
    adv = get_adv(ticker)
    out: list[tuple[str, object]] = [
        ("twap",           TWAPStrategy(num_slices=NUM_SLICES)),
        ("ac_ra",          AlmgrenChrissStrategy(
            num_slices=NUM_SLICES, eta_bps_per_pct_adv=eta,
            gamma_bps_per_pct_adv=gamma, sigma_bps_per_sqrt_sec=SIGMA_DEFAULT,
            lambda_risk=LAMBDA_RISK, adv_shares=adv,
        )),
        ("vwap_following", VWAPFollowingStrategy(num_slices=NUM_SLICES)),
        ("pov_5pct_cap",   POVStrategy(target_pov=None, cap_pov=0.05)),
        ("toth",           TothStrategy(participation_cap=0.10)),
    ]
    if RL_MODEL_PATH.exists():
        out.append(("rl_v2", RLAgentStrategy(
            model_path=str(RL_MODEL_PATH),
            slice_minutes=60, step_seconds=30, n_steps=120,
            observation_mode="v2", max_action_per_step=0.05,
        )))
    else:
        print(f"  ⚠️  {RL_MODEL_PATH} missing — skipping rl_v2")
    return out


def main():
    reports_dir = ROOT / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    n_strategies = 6 if RL_MODEL_PATH.exists() else 5
    print("=" * 70)
    print(f"Phase D — OOS eval: 60-min RL v2 vs 5 Phase-B baselines")
    print(f"5 tickers × 5 days × {n_strategies} strategies "
          f"= {5 * 5 * n_strategies} backtests")
    print(f"Parent: {PARENT_QTY:,} shares sell, 10:00–11:00 ET, NBBO routing")
    print("=" * 70)

    rows = []
    t0 = time.perf_counter()
    n_done = n_failed = 0

    for ticker in TICKERS:
        for date in DATES:
            try:
                df = load_eq_taq(ticker, date)
                engine = BacktestEngine(ticker, date, market_df=df)
                ctx = engine.market_context(bin_minutes=5)
                ctx["adv_shares"] = get_adv(ticker)
            except Exception as e:
                print(f"  ❌ {ticker} {date}: load failed: {e}")
                continue
            parent = ParentOrder(
                ticker=ticker, date=date, side="sell",
                quantity=PARENT_QTY, start_ns=START_TIME_NS, end_ns=END_TIME_NS,
            )
            for label, strat in make_strategies(ticker):
                try:
                    res = engine.run(parent, strat, market_context=ctx)
                    m = res.metrics
                    rows.append({
                        "ticker": ticker, "date": date, "strategy": label,
                        "is_bps": m.is_bps,
                        "vwap_slip_bps": m.vwap_slip_bps,
                        "eff_spread_bps": m.effective_spread_bps,
                        "markout_60s_bps": m.markout_60s_bps,
                        "price_var": m.price_var,
                        "n_fills": len(res.fills),
                    })
                    n_done += 1
                except Exception as e:
                    n_failed += 1
                    print(f"  ❌ {ticker} {date} {label}: {type(e).__name__}: {str(e)[:80]}")
            print(f"  ✓ {ticker} {date}: {n_done} done, {n_failed} failed")

    elapsed = time.perf_counter() - t0
    print(f"\n📊 Sweep done in {elapsed/60:.1f} min — {n_done} done, {n_failed} failed")

    df_results = pl.DataFrame(rows)
    csv_path = reports_dir / "phase_d_rl_v2.csv"
    df_results.write_csv(csv_path)
    print(f"💾 Wrote {csv_path}")

    # Aggregate per strategy: median across (ticker, date)
    agg = (
        df_results
        .group_by("strategy")
        .agg(
            pl.col("is_bps").median().alias("is_bps"),
            pl.col("is_bps").std().alias("is_std"),
            pl.col("vwap_slip_bps").median().alias("vwap_slip_bps"),
            pl.col("eff_spread_bps").median().alias("eff_spread_bps"),
            pl.col("markout_60s_bps").median().alias("markout_60s_bps"),
            pl.col("price_var").median().alias("price_var"),
            pl.len().alias("n_runs"),
        )
        .sort("is_bps")  # most negative IS first = best for sells
    )

    # Win rate vs TWAP per (ticker, date) — RL beats TWAP if IS_RL < IS_TWAP
    twap_runs = df_results.filter(pl.col("strategy") == "twap").select(
        ["ticker", "date", "is_bps"]
    ).rename({"is_bps": "twap_is"})
    win_rates: dict[str, float] = {}
    for s in df_results["strategy"].unique().to_list():
        if s == "twap":
            continue
        joined = (
            df_results.filter(pl.col("strategy") == s)
            .select(["ticker", "date", "is_bps"])
            .join(twap_runs, on=["ticker", "date"], how="inner")
        )
        wr = (joined["is_bps"] < joined["twap_is"]).mean()
        win_rates[s] = float(wr) if wr is not None else 0.0

    # Markdown report
    md = ["# Phase D — Multi-Hour RL with v2 Microstructure Observation\n"]
    md.append(f"**Generated**: {dt.datetime.now(dt.timezone(dt.timedelta(hours=-5))).isoformat()} (ET)")
    md.append(f"**Sweep**: 5 tickers × 5 days × {n_strategies} strategies "
              f"= {n_done + n_failed} backtests  ({n_done} succeeded / {n_failed} failed)")
    md.append(f"**Parent**: {PARENT_QTY:,} shares sell, 10:00–11:00 ET, NBBO routing")
    md.append(f"**RL model**: 60-min episodes, 30-sec steps, 13-dim obs, "
              f"5-ticker × 4-day train pool, 500k timesteps, action cap 5%\n")

    md.append("## Sign convention\n")
    md.append("- IS / VWAP slip / markout: **negative = better for sell** (sold above arrival)")
    md.append("- Win rate vs TWAP: fraction of (ticker, date) pairs where IS_strategy < IS_TWAP\n")

    md.append("## Median across 5 tickers × 5 days, ranked by IS (best first)\n")
    md.append("| Strategy | n | IS (bps) | IS std | VWAP slip | Eff spread | Markout 60s | Win vs TWAP |")
    md.append("|---|---|---|---|---|---|---|---|")
    for row in agg.iter_rows(named=True):
        s = row["strategy"]
        wr_str = f"{win_rates[s]*100:.0f}%" if s != "twap" else "—"
        md.append(
            f"| {s} | {row['n_runs']} | "
            f"**{row['is_bps']:+.2f}** | {row['is_std']:.1f} | "
            f"{row['vwap_slip_bps']:+.2f} | {row['eff_spread_bps']:.2f} | "
            f"{row['markout_60s_bps']:+.2f} | {wr_str} |"
        )

    # Day 5 only — the OOS verdict
    day5 = df_results.filter(pl.col("date") == "20200117")
    if not day5.is_empty():
        md.append("\n## Day 5 OOS only (the honest verdict)\n")
        d5_agg = (
            day5.group_by("strategy").agg(
                pl.col("is_bps").median().alias("median_is"),
                pl.col("is_bps").std().alias("std_is"),
                pl.len().alias("n"),
            ).sort("median_is")
        )
        md.append("| Strategy | n | Median IS (Day 5) | Std |")
        md.append("|---|---|---|---|")
        for r in d5_agg.iter_rows(named=True):
            md.append(
                f"| {r['strategy']} | {r['n']} | "
                f"**{r['median_is']:+.2f}** | {r['std_is']:.1f} |"
            )

    md.append("\n## Caveats\n")
    md.append("- Single 1-hour window 10:00–11:00 ET. Results may differ near open/close.")
    md.append("- 5 days of data; statistical claims should be read with that in mind.")
    md.append("- σ default 1 bps/√sec; not per-ticker calibrated.")
    md.append("- Fills at mid (no spread cost). Real fills should pay half-spread.\n")

    md_path = reports_dir / "phase_d_rl_v2.md"
    md_path.write_text("\n".join(md))
    print(f"💾 Wrote {md_path}")

    # Distribution figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        strategies = sorted(df_results["strategy"].unique().to_list())
        data = [
            df_results.filter(pl.col("strategy") == s)["is_bps"].to_list()
            for s in strategies
        ]
        fig, ax = plt.subplots(figsize=(11, 5))
        parts = ax.violinplot(data, showmedians=True, widths=0.7)
        for pc in parts["bodies"]:
            pc.set_alpha(0.6)
        ax.set_xticks(range(1, len(strategies) + 1))
        ax.set_xticklabels(strategies, rotation=20, ha="right")
        ax.set_ylabel("IS (bps; lower = better)")
        ax.set_title("Phase D — IS distribution across 5 tickers × 5 days")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig_path = figures_dir / "phase_d_rl_v2_distribution.png"
        plt.savefig(fig_path, dpi=120)
        print(f"💾 Wrote {fig_path}")
    except Exception as e:
        print(f"⚠️  Figure failed: {e}")

    print("\n" + "=" * 70)
    print("Phase D complete — see reports/phase_d_rl_v2.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
